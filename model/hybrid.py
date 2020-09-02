from model.img.resne_t import Resne_t
from model.nlp.bert import BERT 
from torch import nn

class Hybrid(nn.Module):
    def __init__(img_model_name, nlp_model_name, img_neuron=256, nlp_neuron=256):
        super().__init__()
        self.img_model = Resne_t(img_model_name, img_neuron)
        self.nlp_model = BERT(nlp_model_name, nlp_neuron)
        self.out = nn.Linear((img_neuron+nlp_neuron), 2)
    
    def forward(self, img, input_ids, attention_mask):
        img_out = self.img_model(img)
        nlp_out = self.nlp_model(input_ids, attention_mask)
        features = torch.cat((img_out, nlp_out), dim=1)
        out = self.out(features)
        return out

