class Aufruf:
    def __init__(self):
        #################### Parameter ####################
        source = 'Steam_Reviews.csv'    
        source2 = False                 
        sampling = False                 
        save = False                    
        inference_only = False
        seed = 42

        model_name = 'answerdotai/ModernBert-base'#'bert-base-uncased'#'roberta-base'#bert-large-uncased'
        #'microsoft/deberta-v3-base'#'distilroberta-base'#'answerdotai/ModernBert-large'
        max_length = 128
        padding = 'max_length'
        truncation = True

        batch_size = 256
        val_size = 0.1

        epochs = 3
        learning_rate = 5e-5
        num_warmup_steps = 0
        scheduler_type = "linear"
        grad_norm_clip_max = 1.0

        test = ["I love Business Analytics and Statistics", "Or do I"]

        s = Sentiment_Analysis(source, source2, sampling, save,
                               inference_only, seed, model_name, padding,
                               max_length, truncation, epochs, batch_size,
                               learning_rate, grad_norm_clip_max, num_warmup_steps,
                               scheduler_type, val_size, test)
    
    
class Sentiment_Analysis:
    def __init__(self, source, source2, sampling, save, inference_only, seed, model_name, padding, max_length, truncation,
                 epochs, batch_size, learning_rate, grad_norm_clip_max, num_warmup_steps, scheduler_type,
                 val_size, test):

        #################### Importing ####################
        import numpy as np
        import pandas as pd
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        from transformers import AutoTokenizer
        from sklearn.model_selection import train_test_split
        from tqdm import tqdm

        try:
            from google.colab import drive
            import os
            drive.mount('/content/drive')
            os.chdir("/content/drive/My Drive/Colab_Notebooks/")
        except:
            pass

        df = pd.read_csv(source)

        torch.set_float32_matmul_precision('high')

        if seed != False:   #Setting seeds
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        #################### Cleaning ####################
        df = self.o_preprocessing(df, source, pd, np, sampling, save, seed)

        #################### Tokenizing ####################
        token_ids, attention_masks = self.o_token(df, source2, torch, AutoTokenizer,
            tqdm, model_name, padding, max_length, truncation, save)

        #################### Splitting ####################
        train_dataloader, val_dataloader = self.o_splitting(df,
            token_ids, attention_masks, torch, train_test_split, TensorDataset, DataLoader,
            val_size, batch_size,seed)

        #################### Modelling ####################
        model, device, zeit, training_losses, validation_losses, validation_accuracies, true_positives, false_positives, false_negatives= self.o_model(
            source2, inference_only, torch, np, tqdm, model_name, epochs, train_dataloader, val_dataloader,
            learning_rate, grad_norm_clip_max, num_warmup_steps, scheduler_type)

        #################### Accuracy ####################
        validation_losses, validation_accuracies = self.o_accuracy(
            model, np, torch, val_dataloader, device, inference_only, validation_losses, validation_accuracies)

        #################### Metrics ####################
        precisions, recalls, f1_scores = self.o_metrics(np, training_losses, validation_losses, validation_accuracies,
            true_positives, false_positives, false_negatives)

        #################### Saving ####################
        self.o_saving(model, torch, model_name, epochs, batch_size, learning_rate, grad_norm_clip_max,
            max_length, padding, truncation, df, training_losses, validation_losses,
            zeit, validation_accuracies, val_size, precisions, recalls, f1_scores, save)

        #################### Test ####################
        self.o_test(test, model, torch, AutoTokenizer, model_name, padding, max_length, truncation)



    #################### Preprocessing ####################
    def o_preprocessing(self, df_core, source, pd, np, sample_o = False, save_df = False, seed = 0):

        #Checking if it isn't already cleaned
        if source.find('cleaned') > -1:
            if sample_o == False:
                return df_core
            else:
                df = df_core.sample(n=sample_o, random_state=seed)
                return df

        #Balancing & Sampling the Dataset
        if sample_o == 0:
            df_n = df_core[df_core['review_score'] == -1]
            df_p = df_core[df_core['review_score'] == 1]

            df = pd.concat([df_n, df_p.sample(n=1500000, random_state=seed)], ignore_index=True)
        elif sample_o != False:
            df_n = df_core[df_core['review_score'] == -1]
            df_p = df_core[df_core['review_score'] == 1]

            df = df_p.sample(n=sample_o, random_state=seed)          #Data-Sample
            df = pd.concat([df, df_n.sample(n=sample_o, random_state=seed)], ignore_index=True)
        else:
            df = df_core

        #Sentiment von -1 und 1 zu 0 und 1 kodieren
        df['review_score'] = df['review_score'].\
            apply(lambda x: 0 if x == -1 else 1)

        #Floats to Strings
        df['review_cleaned'] = [str(w) for w in df['review_text']]

        #Kürzen auf relevante Spalten
        df = df[['review_cleaned','review_score']]

        # -----Kürzungen:-----
        # Unnötige Leerzeichen entfernen
        df['review_cleaned'] = df['review_cleaned'].\
            replace(r'\s+', ' ', regex=True)

        #Links entfernen
        df['review_cleaned'] = df['review_cleaned'].\
            replace(r'http\S+', '', regex=True)
        df['review_cleaned'] = df['review_cleaned'].\
            replace(r'www\S+', '', regex=True)

        #10/10 und 0/10 ersetzen -- hier könnte man noch mehr machen mit regex oder bereich an zahlen
        df['review_cleaned'] = df['review_cleaned'].\
            replace("10/10", 'perfect', regex=False)
        df['review_cleaned'] = df['review_cleaned'].\
            replace("0/10", 'horrible', regex=False)

        #Nummern entfernen
        df['review_cleaned'] = df['review_cleaned'].\
            replace(r'\d+', '', regex=True)

        #Symbole entfernen, außer ". , !"
        df['review_cleaned'] = df['review_cleaned'].\
            replace(r'[^\w\s.,!]', '', regex=True)

        #Mehrere Punkte hintereinander durch einen Punkt ersetzen
        df['review_cleaned'] = df['review_cleaned'].str.replace(r'\.+', '.', regex=True)

        #Mehrere Ausrufezeichen hintereinander durch einen Punkt ersetzen
        df['review_cleaned'] = df['review_cleaned'].str.replace(r'\!+', '!', regex=True)

        #Early-Acess Reviews entfernen (fehlerhaftes Scraping)
        df = df.drop(df.index[df["review_cleaned"] == " Early Access Review"])

        #nur Leerzeichen zu NaN machen
        df['review_cleaned'] = df['review_cleaned'].\
            replace(r'^\s*$', np.nan, regex=True)

        #NaN Reviews entfernen
        df = df.drop(df.index[df["review_cleaned"] == "nan"])

        #Missing Values entfernen
        df = df.dropna(subset=['review_cleaned'])

        #Entfernen von Spalten mit nur "."
        df = df.drop(df.index[df["review_cleaned"] == "."])

        if sample_o == 0:

            #Erzeuge gleich viele positive wie negative Reviews
            count_n = df['review_score'].value_counts()[0]
            df = pd.concat([df[df['review_score'] == 1].sample(n=count_n, random_state=seed),
                            df[df['review_score'] == 0]], ignore_index=True)

        if save_df == 1 or save_df == -1:
            df.to_csv('cleaned_reviews.csv', index=False)  #Speichern

        print(f'Number of reviews: {df.value_counts("review_score")}')

        return df



    #################### Tokenizing ####################
    def o_token(self, df, source2, torch, AutoTokenizer, tqdm, model_name = 'answerdotai/ModernBERT-base',
                    padding = 'max_length', max_length = 128, truncation = True, save = False):

        #First to check, whether we have given Tokens in source2
        if source2 != False and source2.find('.pth') == -1:
            df_t = torch.load(source2)
            token_ids = df_t[0]
            attention_masks = df_t[1]
            del df_t
            return token_ids, attention_masks

        #Our tokenizer, depending on the name of the model
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        #We put the tokenizer in a for loop to get the tqdm-progress-Bar, else .map
        tokenized_df  = [tokenizer(i,
                    padding = padding,
                    max_length = max_length,
                    truncation = truncation,
                    return_tensors = 'pt'
                    ) for i in tqdm(df['review_cleaned'], desc="Tokenizing", unit="text")]

        #Taking the values out of the Batch-Encodes in the Lists and pack it into a torch-Tensor
        token_ids = torch.cat([i['input_ids'] for i in tokenized_df], dim=0)
        attention_masks = torch.cat([i['attention_mask'] for i in tokenized_df], dim=0)

        #Saving as a torch Object
        if save == 2 or save == -1:
            torch.save((token_ids, attention_masks), 'tokens.pt')

        return token_ids, attention_masks



    #################### Splitting ####################
    def o_splitting(self, df, token_ids, attention_masks, torch, train_test_split, TensorDataset,
                    DataLoader, val_size=0.1, batch_size=64,seed=0):

        # Split the token IDs
        train_ids, val_ids = train_test_split(
                                token_ids,
                                test_size=val_size,
                                shuffle=True,
                                random_state=seed)

        # Split the attention masks
        train_masks, val_masks = train_test_split(
                                    attention_masks,
                                    test_size=val_size,
                                    shuffle=True,
                                    random_state=seed)

        # Split the labels
        labels = torch.tensor(df['review_score'].values)
        train_labels, val_labels = train_test_split(
                                        labels,
                                        test_size=val_size,
                                        shuffle=True,
                                        random_state=seed)

        # Create the DataLoaders
        train_data = TensorDataset(train_ids, train_masks, train_labels)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=0)
        val_data = TensorDataset(val_ids, val_masks, val_labels)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=0)

        return train_dataloader, val_dataloader



    #################### Modelling ####################
    def o_model(self, source2, inference_only, torch,np, tqdm, model_name, epochs, train_dataloader, val_dataloader,
                learning_rate, grad_norm_clip_max, num_warmup_steps=0, scheduler_type = "linear"):

        from torch.optim import AdamW
        from transformers import AutoModelForSequenceClassification, get_scheduler
        import time
        from transformers import ModernBertConfig, ModernBertForSequenceClassification

        #configuration = ModernBertConfig()

        #model = ModernBertForSequenceClassification(model_name,configuration=configuration, num_labels=2)


        #pre-trained Model initialisation with 2 labels
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # return model
        if source2 != False and source2.find('.pth') > -1:
            model.load_state_dict(torch.load(source2, weights_only=True))

        # Check if GPU is available for faster training time
        if torch.cuda.is_available():
            print(f"CUDA Available: {torch.cuda.is_available()}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda')     #cuda":0" -> first GPU (standard anyway)
            print("GPU")
        else:
            device = torch.device('cpu')
            print("CPU")

        if inference_only:
            return model, device, 0, 0, 0, 0

        # Optimizer: AdamW most effective, regulates weight decay well. Alternative: AdaFactor
        optimizer = AdamW(model.parameters(),lr=learning_rate) #lr: Initial (Start) learning rate

        # Scheduler, adapts the learning rate, so the rate isn't fixed
        num_training_steps = epochs * len(train_dataloader) #Amount of Training Steps

        scheduler = get_scheduler(
            scheduler_type,     #Linear, Constant or f.e. Cosine, influecens how the changes occur
            optimizer,
            num_warmup_steps=num_warmup_steps,      #for Linear: Warumup: slowly increase until the given step amount
            num_training_steps=num_training_steps)  #for Linear: After Warumup, it only decreases

        #Training-Process
        start_t = time.time()   #Timer
        model.to(device)        #Moving Model to GPU
        training_losses, validation_losses, validation_accuracies= [],[],[]
        true_positives, false_positives, false_negatives = [],[],[]

        for epoch in range(0, epochs):  #Epochs-loop

            model.train()   #Set Model into Training-Mode, in this mode the model behaves differently.
                            #this will f. e. active specific layers for training

            training_loss = 0   #Training-loss per Epoch, it should decrese per epoch

            for batch in tqdm(train_dataloader):    #Extracts every row with tokens etc. from the dataloader

                batch_token_ids = batch[0].to(device)   #Moving to GPU
                batch_attention_mask = batch[1].to(device)
                batch_labels = batch[2].to(device)

                model.zero_grad()       #Sets gradients, used in Backpropagation, to zero, else they would sum up
                                        #Same as optimizer.zero_grad()

                outputs = model(   #Puts the data into the model, returns loss
                    batch_token_ids,    #model(**batch) would be a more compact variant
                    attention_mask=batch_attention_mask,
                    labels=batch_labels,
                    return_dict=True)   #Return either Outputs or loss & logits directly
                                        #Logits would be the predictions based on the current classifier weights (& biases)
                                        #Therefore they will first be of interest in the evaluation, in accuracy

                loss = outputs.loss             #Loss is calculated by the loss function
                training_loss += loss.item()    #Adds this loss to the Training-loss per Epoch, only relevant for afterwards

                loss.backward()     #dloss/dx, if we had multiple losses, we could sum them up before doing this
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip_max)
                                    #Gradient normalisation for stabilisation (exploding gradients problem)
                optimizer.step()    #Updates Model-Parameters, how is depending on the optimizer and the loss
                scheduler.step()    #Adjusts the learning rate


            average_train_loss = training_loss / len(train_dataloader)  #All calculated losses summed up then mean'ed
            training_losses.append(average_train_loss)

            model.eval()    #Setting the model in evaluation mode

            #Metric-Lists
            val_accuracy,val_loss = 0,0
            true_positive, false_negative, false_positive = 0,0,0

            for batch in val_dataloader:

                batch_token_ids = batch[0].to(device)   #To GPU, doesnt really matter tho
                batch_attention_mask = batch[1].to(device)
                batch_labels = batch[2].to(device)

                with torch.no_grad():   #Only for evaluation
                    (loss, logits) = model( #Return loss and logits for test-dataset
                        batch_token_ids,
                        attention_mask = batch_attention_mask,
                        labels = batch_labels,
                        return_dict=False)

                logits = logits.detach().cpu().numpy()  #.numpy() hast to be on cpu
                label_ids = batch_labels.to('cpu').numpy()
                val_loss += loss.item()    # adss the loss to the val-losses for the current epoch
                metrics = self.calculate_accuracy(np, logits, label_ids) #calculates metrics
                val_accuracy += metrics[0]
                true_positive += metrics[1]
                false_positive += metrics[2]
                false_negative += metrics[3]


            average_val_accuracy = val_accuracy / len(val_dataloader)       #making averages
            average_val_loss = val_loss / len(val_dataloader)

            #Appending to Metric-Lists
            true_positives.append(int(true_positive))
            false_positives.append(int(false_positive))
            false_negatives.append(int(false_negative))
            validation_losses.append(float(average_val_loss))
            validation_accuracies.append(float(average_val_accuracy))

        zeit = time.time() - start_t    #Time the model took to process & evaluate

        return model, device, zeit, training_losses, validation_losses, validation_accuracies, true_positives, false_positives, false_negatives

    def calculate_accuracy(self, np, preds, labels):  #Rakes predictions and their corresponding labels
        pred_flat = np.argmax(preds, axis=1).flatten() #The predictions are logits and must first be put into normal numbers
        accuracy = np.sum(pred_flat == labels) / len(labels)  #Returns percentile accuracy
        true_positives = np.sum((pred_flat == 1) & (labels == 1))   #Needed for recall etc.
        false_positives = np.sum((pred_flat == 1) & (labels == 0))
        false_negatives = np.sum((pred_flat == 0) & (labels == 1))
        return [accuracy, true_positives, false_positives, false_negatives]



    #################### Accuracy ####################
    def o_accuracy(self, model, np, torch, val_dataloader, device, inference_only, average_val_loss, average_val_accuracy):

        if inference_only == False: #o_accuracy is only needed if inference on an already finished model is wished
            return average_val_loss, average_val_accuracy

        model.eval()    #Setting the model in evaluation mode
        val_loss = 0
        val_accuracy = 0

        for batch in val_dataloader:

            batch_token_ids = batch[0].to(device)   #To GPU, doesnt really matter tho
            batch_attention_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)

            with torch.inference_mode():   #Only for inference
                (loss, logits) = model( #Return loss and logits for test-dataset
                    batch_token_ids,
                    attention_mask = batch_attention_mask,
                    labels = batch_labels,
                    return_dict=False)

            logits = logits.detach().cpu().numpy()  #.numpy() muss auf cpu sein
            label_ids = batch_labels.to('cpu').numpy()
            val_loss += loss.item()
            val_accuracy += self.calculate_accuracy(np, logits, label_ids)

        average_val_accuracy = val_accuracy / len(val_dataloader)
        average_val_loss = val_loss / len(val_dataloader)

        return average_val_loss, average_val_accuracy

    def o_metrics(self, np, training_losses, validation_losses, validation_accuracies,
              true_positives, false_positives, false_negatives):

        #Calculation for Metrics per Epoch
        precisions, recalls, f1_scores = [],[],[]
        for i in range(len(true_positives)):
            precision = (true_positives[i] / (true_positives[i] + false_positives[i])) #nan if equation doens't work
            recall = true_positives[i] / (true_positives[i] + false_negatives[i])
            f1_score = 2 * (precision * recall) / (precision + recall)      #f1-score calculation

            precisions.append(precision)    #Appending per to per epoch result lists
            recalls.append(recall)
            f1_scores.append(f1_score)

            #Calculating true_negatives
            true_negatives = (validation_accuracies[i]*(false_positives[i] + false_negatives[i]))/(1-validation_accuracies[i]) - true_positives[i]
            true_negatives = round(true_negatives)

            #Confusion-Matrix
            self.o_conf_matrix(i, true_positives[i], false_positives[i], false_negatives[i], true_negatives)

        return precisions, recalls, f1_scores

    def o_conf_matrix(self, i,true_positives, false_positives, false_negatives, true_negatives):

        from tabulate import tabulate

        table = [["Confusion-Matrix-{}".format(i+1),"Actually Positive","Actually Negative"],
                ["Predicted Positive", true_positives, false_positives],
                ["Predicted Negative", false_negatives, true_negatives]]

        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))



    #################### Saving ####################
    def o_saving(self, model, torch, model_name, epochs, batch_size, learning_rate, grad_norm_clipping,
            max_length, padding, truncation, df, average_train_loss, average_val_loss,
            zeit, average_val_accuracy, val_size, precisions, recalls, f1_scores, save):

        if save == 3 or save == -1:
            torch.save(model.state_dict(), "my_model.pth")

        if zeit == -1:
            return

        ausgabe = [""]

        with open("data_log.txt","a") as dl:
            ausgabe = [
            f"Model: {model_name}\n",
            f"Optimizer: {"AdamW"}\n",
            f"Size: {len(df)}\n",
            f"Time: {zeit}\n",
            f"Epochs: {epochs}\n",
            f"Batch Size: {batch_size}\n",
            f"Learning Rate: {learning_rate}\n",
            f"Gradient Norm Clipping: {grad_norm_clipping}\n",
            f"Max Length: {max_length}\n",
            f"Padding: {padding}\n",
            f"Truncation: {truncation}\n",
            f"Validation Size: {val_size}\n",
            f"Validation Loss: {[i for i in average_val_loss]}\n",
            f"Training Loss: {[i for i in average_train_loss]}\n",
            f"Precision: {[i for i in precisions]}\n",
            f"Recall: {[i for i in recalls]}\n",
            f"F1-Score: {[i for i in f1_scores]}\n",
            f"Validation Accuracy: {[i for i in average_val_accuracy]}\n",
            "--------------------\n"]
            [dl.write(i) for i in ausgabe]



    #################### Test ####################
    def o_test(self, texts, model, torch, tokenize_model, model_name,
                    padding = 'max_length', max_length = 128, truncation = True):

        tokenizer = tokenize_model.from_pretrained(model_name) #Tokenize
        model.to("cpu") #small enough for cpu

        # Encode each text
        for review in texts:
            batch_encoded = tokenizer.encode_plus(  #Deducted way to tokenzie
                review,
                max_length = True,
                padding = False,
                truncation = truncation,
                return_tensors = 'pt')

            #o_show_tokens(review,batch_encoded.input_ids, tokenizer)

            with torch.no_grad():
                logits = model(**batch_encoded).logits  #Simpler way to call the model

            probs = torch.softmax(logits, dim=-1)
            score = probs[0, 1].item()
            ausgabe = "Positive Sentiment" if score >= 0.5 else "Negatives Sentiment"
            print(score, ausgabe)

    def o_show_tokens(self, text,batch_encoded, tokenizer):
        for each in batch_encoded:
            print()
            print(text) #text
            print(tokenizer.batch_decode(batch_encoded))    #Text mit cls
            print(tokenizer.convert_ids_to_tokens(each)) #Text getrimmt
            print(batch_encoded.tolist())   #ids

            
A = Aufruf()