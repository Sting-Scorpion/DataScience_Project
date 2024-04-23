import torch
import os
from flask import Flask, request, render_template
from model import InitializeClassifier, tokenize

app = Flask(__name__)
# Select device
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = '../pretrained/bert-base-uncased'
assert os.path.exists(model_path)
weight_path = '../model/Bert_classifier.pth'
assert os.path.exists(weight_path)

# Load Model
model = InitializeClassifier(model_path, weight_path)

model.eval()
# labels info
class_labels = {0: 'non-suicide', 1: 'suicide'}

@app.route('/')
def root():
    return render_template('page.html',
                           input_text_content='',
                           alert_display_area='hidden',
                           normal_display_area='hidden')

# /predict for prediction
@app.route('/predict', methods=['POST', 'GET'])
@torch.no_grad()
def predict():
    if request.method == 'POST':
        text = request.form.get('input_text')

        sen_ids, attention_masks = tokenize(text, model_path)

        output = model(input_ids=sen_ids, attention_mask=attention_masks, token_type_ids=None)
        pred = output['logits']
        probability = torch.nn.functional.softmax(pred, dim=1)
        probability = probability.numpy().squeeze()
        class_index = probability.argmax()

        proda = '{:.2%}'.format(probability[1])
        preda = '{}'.format(class_labels[class_index])
        alertda = 'hidden'
        display_color = 'rgb(99, 141, 220)'
        border_color = 'rgb(39, 82, 161)'
        if class_index == 1:
            alertda = 'visible'
            display_color = 'rgb(220, 99, 99)'
            border_color = 'rgb(161, 39, 39)'

        return render_template('page.html', 
                               probability_display_area=proda,
                               prediction_display_area=preda,
                               alert_display_area=alertda,
                               normal_display_area='visible',
                               d_color=display_color,
                               b_color=border_color,
                               input_text_content=text)

if __name__ == '__main__':
    app.run(debug=True)
