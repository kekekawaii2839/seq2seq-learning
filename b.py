from data import *
from torch import optim

hidden_size = 512
n_layers = 3
dropout_p = 0.05
learning_rate = 1e-6

# Initialize models
model = myTransformer(input_lang.n_words, hidden_size, output_lang.n_words, n_layers, 8, dropout_p=dropout_p)
criterion = nn.CrossEntropyLoss()

# Move models to GPU
if USE_CUDA:
    model.cuda()
print(all(p.is_cuda for p in model.parameters()))

# Initialize optimizers and criterion
model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Configuring training
n_epochs = 100000
print_every = 1000
save_every = 2000

# Keep track of time elapsed and running averages
start = time.time()
print_loss_total = 0 # Reset every print_every

print(torch.cuda.get_device_name(0))

def evaluate(sentence):
    input_variable = variable_from_sentence(input_lang, sentence)
    init = torch.LongTensor([[SOS_token]])
    if USE_CUDA:
        init = init.cuda()
    for i in range(MAX_LENGTH+1):
        output = model(input_variable, init)
        prob = model.out(output[-1,0,:])
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.item()
        init = torch.cat([init, torch.LongTensor([[next_word]]).cuda()], dim=0)
        if next_word == EOS_token:
            break
    output_sentence = [output_lang.index2word[int(i)] for i in init]
    output_sentence = ' '.join(output_sentence)

    return output_sentence

def evaluate_randomly():
    [input_sentence, target_sentence] = random.choice(pairs)
    output_sentence = evaluate(input_sentence)
    print('>', input_sentence)
    print('=', target_sentence)
    print('<', output_sentence)
    print('')

# Begin!
if os.path.exists('model.pth'):
    model.load_state_dict(torch.load('model.pth'))
    print('model loaded')
    for i in range(10):
        evaluate_randomly()
    flag = input('train? (y/n)')
    if flag == 'n':
        n_epochs = 0
else:
    print('model not found')

for epoch in range(1, n_epochs + 1):
    # Get training data for this cycle
    [input_sentence, target_sentence] = random.choice(pairs)
    input_variable = variable_from_sentence(input_lang, input_sentence)
    target_variable = variable_from_sentence(output_lang, target_sentence)

    #print("input_variable.shape:", input_variable.shape)
    #print("target_variable.shape:", target_variable.shape)
    
    # Run the train function
    loss = train(input_variable, target_variable, model, model_optimizer, criterion)
    print_loss_total += loss
    
    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)
        model.eval()
        with torch.no_grad():
            evaluate_randomly()
        model.train()
    
    if epoch % save_every == 0:
        torch.save(model.state_dict(), 'model.pth')
        print('model saved')

torch.save(model.state_dict(), 'model.pth')

model.eval()
with torch.no_grad():
    for i in range(10):
        evaluate_randomly()
a = input('Chinese sentence: ')
while a != 'exit':
    print(evaluate(a))
    a = input('Chinese sentence: ')