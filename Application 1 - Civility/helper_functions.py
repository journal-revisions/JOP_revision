import seaborn as sns
import matplotlib.pyplot as plt
import datetime 
import time
import torch
import numpy as np

def good_update_interval(total_iters, num_desired_updates):
    exact_interval = total_iters / num_desired_updates
    order_of_mag = len(str(total_iters)) - 1
    round_mag = order_of_mag - 1
    update_interval = int(round(exact_interval, -round_mag))
    if update_interval == 0:
        update_interval = 1
    return update_interval
    
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded)) 

def plot_distribution(list_lengths):
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (10,5)
    trunc_lengths = [min(l, 512) for l in list_lengths]
    sns.distplot(trunc_lengths, kde=False, rug=False)
    plt.title('Comment Lengths')
    plt.xlabel('Comment Length')
    plt.ylabel('# of Comments')
    
def plot_loss(df):
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)
    plt.plot(df['Training Loss'], 'b-o', label="Training")
    plt.plot(df['Valid. Loss'], 'g-o', label="Validation")
    plt.title("Training & Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(list(range(1, epochs+1)))
    plt.show()


def plot_loss_cv(df, epochs):
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)
    plt.plot(df.iloc[0,:], 'b-o', label="Training")
    plt.plot(df.iloc[1,:], 'g-o', label="Validation")
    plt.title("Training & Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(list(range(1, epochs+1)))
    plt.show()





def train(model, epochs, train_dataloader, validation_dataloader, device, optimizer, scheduler):
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        model.train()
        update_interval = good_update_interval(
                    total_iters = len(train_dataloader),
                    num_desired_updates = 10
                )
                
        for step, batch in enumerate(train_dataloader):
    
            if (step % update_interval) == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)[0]
            logits = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)[1]
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        
        print("")
        print("Running Validation...")
        t0 = time.time()
        model.eval()
        total_eval_loss = 0
        predictions, true_labels = [], []
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
                loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)[0]
                logits = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)[1]
            
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.append(logits)
            true_labels.append(label_ids)
        
        flat_predictions = np.concatenate(predictions, axis=0)
        flat_true_labels = np.concatenate(true_labels, axis=0)
        predicted_labels = np.argmax(flat_predictions, axis=1).flatten()
        val_accuracy = (predicted_labels == flat_true_labels).mean()
        zeroes = np.where(flat_true_labels == 0)
        ones = np.where(flat_true_labels == 1)
        zero_acc = len(predicted_labels[zeroes][predicted_labels[zeroes]==0])/len(predicted_labels[zeroes])
        one_acc = len(predicted_labels[ones][predicted_labels[ones]==1])/len(predicted_labels[ones])
        
        print("  0: {0:.2f}".format(zero_acc))
        print("  1: {0:.2f}".format(one_acc))
        print('RoBERTa Prediction accuracy: {:.3f}'.format(val_accuracy))
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
    
 
    
    
    