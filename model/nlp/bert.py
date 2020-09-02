from transformers import BertModel
from torch import nn

class BERT(nn.Module):
    def __init__(model_model, num_neuron=256):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(nlp_model_name)
        self.num_neuron = num_neuron
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size, self.num_neuron)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert_model(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)
