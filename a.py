from data import *
from torch import optim

# .pth

hidden_size = 512
n_layers = 4
dropout_p = 0.05

# Initialize models
encoder = Encoder(input_lang.n_words, hidden_size, n_layers)
decoder = Decoder(2*hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)
#criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
model = Seq2Seq(encoder, decoder, criterion)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

print(all(p.is_cuda for p in encoder.parameters()))
print(all(p.is_cuda for p in decoder.parameters()))

# Initialize optimizers and criterion
learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

# Configuring training
n_epochs = 100000
print_every = 1000
save_every = 2000

# Keep track of time elapsed and running averages
start = time.time()
print_loss_total = 0 # Reset every print_every

print(torch.cuda.get_device_name(0))

def evaluate(sentence, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    
    # Run through encoder
    encoder_hidden = model.encoder.init_hidden()
    encoder_outputs, encoder_hidden = model.encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([[SOS_token]]) # SOS
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden
    decoder_hidden = [i.view(model.decoder.n_layers, 1, -1) for i in encoder_hidden]
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni.item()])
            
        # Next input is chosen word
        decoder_input = torch.LongTensor([[ni]])
        if USE_CUDA: decoder_input = decoder_input.cuda()
    
    return decoded_words

def evaluate_randomly():
    pair = random.choice(pairs)
    
    output_words = evaluate(pair[0])
    output_sentence = ' '.join(output_words)
    
    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')

# Begin!
if os.path.exists('model.pth'):
    model.load_state_dict(torch.load('model.pth'))
    print('model loaded')
    for i in range(100):
        evaluate_randomly()
    flag = input('train? (y/n)')
    if flag == 'n':
        n_epochs = 0
else:
    print('model not found')

for epoch in range(1, n_epochs + 1):
    # Get training data for this cycle
    training_pair = variables_from_pair(random.choice(pairs))
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    if USE_CUDA:
        input_variable = input_variable.cuda()
        target_variable = target_variable.cuda()

    # Run the train function
    loss = train(input_variable, target_variable, model, encoder_optimizer, decoder_optimizer)

    # Keep track of loss
    print_loss_total += loss
    if epoch == 0: continue
    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)
    
    if epoch % save_every == 0:
        print('saving model')
        torch.save(model.state_dict(), 'model.pth')
        print('model saved, epoch =', epoch)
        print("evaluate test: ")
        evaluate_randomly()

torch.save(model.state_dict(), 'model.pth')

for i in range(100):
    evaluate_randomly()