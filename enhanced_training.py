#!/usr/bin/env python3
"""
Enhanced Training Script for Multilingual LLM
Improved training with more sophisticated techniques and larger dataset
"""

import os
import json
import time
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Enhanced dataset creation
def create_enhanced_dataset():
    """Create a much larger, more comprehensive trilingual dataset"""
    
    enhanced_data = [
        # Basic greetings and conversations
        "Hello, how are you today? I hope you are doing well.",
        "Habari yako leo? Natumai uko mzuri.",
        "Wĩ atĩa ũmũthĩ? Nĩndĩramaini ũguo mwega.",
        "Inadi kawuono? Ageno ni ber.",
        
        # Love and relationships
        "I love you so much, you mean everything to me.",
        "Nakupenda sana, wewe ni kila kitu kwangu.",
        "Nĩngũkwenda mũno, wee nĩwe ũndũ wothe harĩ niĩ.",
        "Aheri miwuoro matek, in e gima duto e ngimana.",
        
        # Family and relationships
        "My family is very important to me. We love each other deeply.",
        "Familia yangu ni muhimu sana kwangu. Tunapendana sana.",
        "Nyũmba yakwa nĩ ya bata mũno harĩ niĩ. Twendanĩtie mũno.",
        "Jooda en gima ber ahinya e ngimana. Waheruore adier.",
        
        # Daily life and activities
        "Today I went to the market to buy food for my family.",
        "Leo nilienda sokoni kununua chakula cha familia yangu.",
        "Ũmũthĩ ndathiire ithũa kũgũra irio cia nyũmba yakwa.",
        "Kawuono nadhi chiro mondo angʼiew chiemo ne jooda.",
        
        # Weather and nature
        "The weather is beautiful today. The sun is shining brightly.",
        "Hali ya anga ni nzuri leo. Jua linaangaza kwa nguvu.",
        "Mũthĩ wa rĩu mwega mũno. Riũa nĩrĩrahe wega.",
        "Kinde nigi ber kawuono. Wangʼ rieny maber.",
        
        # Education and learning
        "Education is very important for our children's future.",
        "Elimu ni muhimu sana kwa maisha ya watoto wetu.",
        "Gũthoma nĩ kũrĩ bata mũno nĩ ũndũ wa thutha wa ciana ciitũ.",
        "Puonjruok ber ahinya ne kinde mabiro mar nyithindwa.",
        
        # Work and business
        "I work hard every day to provide for my family.",
        "Ninafanya kazi kwa bidii kila siku kulipa gharama za familia.",
        "Ndũthomaga na hinya o mũthenya nĩgeetha ndũteithagie nyũmba yakwa.",
        "Atiyo matek pile ka pile mondo akonyre jooda.",
        
        # Food and cooking
        "My mother cooks the most delicious traditional food.",
        "Mama yangu anapika chakula cha jadi chenye ladha sana.",
        "Maaitũ arugaga irio cia gĩkũyũ iria cĩ mũruru mũno.",
        "Mama wuod chiemo mar jo-Luo ma mit ahinya.",
        
        # Travel and places
        "I want to visit different places and meet new people.",
        "Nataka kutembelea maeneo tofauti na kukutana na watu wapya.",
        "Nĩngwenda gũthiĩ kũndũ kũngĩ na ndutane na andũ angĩ.",
        "Adwaro dhi kuonde mopogore gi romo gi ji manyien.",
        
        # Health and wellbeing
        "It's important to eat healthy food and exercise regularly.",
        "Ni muhimu kula chakula chenye afya na kufanya mazoezi.",
        "Nĩ bata kũrĩa irio cĩ ũgima na gwĩka ciĩko cia mwĩrĩ.",
        "En gima ber chamo chiemo maber kod timo ka maber.",
        
        # Technology and modern life
        "Technology has changed how we communicate with each other.",
        "Teknolojia imebadilisha jinsi tunavyoongea na wenzetu.",
        "Ũmenyi wa indo nĩũgarũrĩte ũrĩa twaragania nao andũ angĩ.",
        "Rieko mar lwete oloko kaka wawuoyore.",
        
        # Culture and traditions
        "Our traditional customs are important to preserve for future generations.",
        "Desturi zetu za jadi ni muhimu kuzihifadhi kwa vizazi vijavyo.",
        "Mĩtugo iitũ ya tene nĩ ya bata kũmĩiga nĩ ũndũ wa andũ arĩa magooka.",
        "Timbe mag-wa machon ber mako ne tienge mabiro.",
        
        # Hope and aspirations
        "I hope that one day all children will have access to good education.",
        "Natumai siku moja watoto wote watapata elimu nzuri.",
        "Nĩndĩramaini atĩ mũthenya ũmwe ciana ciothe igetha gũthoma.",
        "Ageno ni odiechieng moro nyithindo duto noyud puonjruok maber.",
        
        # Community and togetherness
        "When we work together, we can achieve great things.",
        "Tunapofanya kazi pamoja, tunaweza kufanikisha mambo makubwa.",
        "Rĩrĩa tũrutanagĩra wĩra-rĩ, no tũhinge tũndũ nene.",
        "Ka wa timo gimoro kanyakla to wanyalo hingio gik madongo.",
        
        # Wisdom and advice
        "Listen to your elders, they have much wisdom to share.",
        "Sikiliza wazee wako, wana busara nyingi za kushiriki.",
        "Thikĩrĩria athuuri aku, nĩ marĩ na ũũgĩ mũingĩ wa kũheana.",
        "Winj jodongo, gin gi rieko mangʼeny me miyo ji.",
        
        # Conversations and storytelling
        "Tell me a story about the old days in our village.",
        "Niambie hadithi kuhusu siku za zamani katika kijiji chetu.",
        "Njĩra rũgano rwa matukũ ma tene kũu itũũra-inĩ riitũ.",
        "Nyisa sigana kuom kinde machon e gwengwa.",
        
        # Questions and responses
        "What do you think about this? I would like to hear your opinion.",
        "Unafikiri nini kuhusu hii? Ningependa kusikia maoni yako.",
        "Wĩciiria atĩa ũhoro-inĩ ũyũ? Nĩngwenda kũigua mũyũkaniu waku.",
        "Iparo nango kuom wachni? Adwaro winjo pachi.",
        
        # Complex conversations about life
        "Life is full of challenges, but we must remain strong and hopeful.",
        "Maisha yamejaa changamoto, lakini lazima tubaki imara na wenye matumaini.",
        "Muoyo nĩũiyũrĩte mathĩĩna, no nĩtũgwĩrĩire gũtũũra tũrĩ na hinya na mwĩhoko.",
        "Ngima opongʼ gi chandruok, to nyaka wasiki motegno kod kinde mar geno.",
        
        # Advanced language patterns
        "The beauty of our languages lies in their ability to express deep emotions.",
        "Uzuri wa lugha zetu upo katika uwezo wake wa kuelezea hisia za kina.",
        "Ũthaka wa lũthiomi rwaitũ ũrĩ atĩ nĩrũhota kũguũria meciiria ma thĩinĩ.",
        "Ber mar dhogewa nitie kuom teko mare mar nyiso chir ma otut.",
        
        # Mixed language conversations (code-switching)
        "Hello habari, how are you leo?",
        "Nimefika home, nĩndĩrakũmenyera.",
        "Good morning, wuon angʼo makoro?",
        
        # Longer conversational sequences
        "Person A: Hello, how has your day been?\nPerson B: It has been good, thank you. I went to work and then visited my family.\nPerson A: That sounds wonderful. Family time is always precious.",
        
        "Person A: Habari za leo?\nPerson B: Nzuri sana, asante. Nimekuwa kazi na baadaye nimetembelea familia.\nPerson A: Hiyo ni vizuri sana. Wakati wa familia ni muhimu.",
        
        "Person A: Wĩ atĩa ũmũthĩ?\nPerson B: Nĩnguo mwega, nĩ ũngĩ. Nĩndĩrathiira wĩra na thuutha nĩndĩrakora andũ a nyũmba.\nPerson A: Ũguo nĩ gwega mũno. Ihinda rĩa nyũmba nĩ rĩa bata.",
        
        "Person A: Inadi kawuono?\nPerson B: Aber ahinya, erokamano. Asetiyo kendo bang ame aneno jooda.\nPerson A: Mano ber ahinya. Kinde mar jooda en gima ber.",
        
        # Emotional expressions
        "I am so happy to see you today! You bring joy to my heart.",
        "Nimefurahia sana kukuona leo! Unaleta furaha moyoni mwangu.",
        "Nĩndĩrakena mũno gũkuona ũmũthĩ! Ũrehere gĩkeno ngoro-inĩ yakwa.",
        "Amor ahinya ne neno kawuono! Ikelo mor e chunya.",
        
        # Describing actions and activities
        "Yesterday I was walking in the garden and I saw beautiful flowers blooming.",
        "Jana nilikuwa nikitembea bustanini na nikaona maua mazuri yakichipua.",
        "Ira ndaarĩ njĩra mũgũnda-inĩ na ngĩona mathanju mathaka makĩaruthanĩra.",
        "Nyoro ne awuotho e puothe kendo aneno omodho maber mag thidhoya.",
        
        # Weather and seasons
        "During the rainy season, everything becomes green and beautiful.",
        "Wakati wa mvua, kila kitu kinakuwa kijani na kizuri.",
        "Hĩndĩ ya mbura-rĩ, ũndũ wothe ũtuĩkaga mũruru na mũthaka.",
        "E kinde mar kok, gik moko duto bedo raum kod ber.",
        
        # Complex narratives
        "Once upon a time, there was a wise old man who lived in the mountains. People would come from far and wide to seek his advice because he had seen many things in his long life.",
        
        "Hapo zamani, kulikuwa na mzee mwenye busara aliyeishi milimani. Watu walikuja kutoka mbali ili kupata ushauri wake kwa sababu alikuwa ameona mambo mengi katika maisha yake marefu.",
        
        "Tene ho kũũrĩ na mũthuuri mũũgĩ ũrĩa watuĩraga irĩma-inĩ. Andũ mookaga kuuma kũraya nĩguo maheo kĩgaanĩro nĩ ũndũ nĩakoragwo onete maũndũ maingĩ muoyo-inĩ wake mũraihu.",
        
        "Chon nolal jaduongʼ ma nigi rieko mane odak e gode. Ji biro koa mabor mondo giyud rieko nikech nosewinjo gik mangʼeny e ngimane mawielo."
    ]
    
    return enhanced_data

class EnhancedTextDataset(Dataset):
    """Enhanced dataset with better preprocessing and data augmentation"""
    
    def __init__(self, texts: List[str], max_length: int = 512, tokenizer=None):
        self.texts = texts
        self.max_length = max_length
        
        # Create a simple character-level tokenizer if none provided
        if tokenizer is None:
            self.tokenizer = self._create_tokenizer()
        else:
            self.tokenizer = tokenizer
    
    def _create_tokenizer(self):
        """Create a character-level tokenizer with special tokens"""
        all_text = ' '.join(self.texts)
        unique_chars = sorted(list(set(all_text)))
        
        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>']
        vocab = special_tokens + unique_chars
        
        char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        
        return {
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'vocab_size': len(vocab),
            'pad_token_id': 0,
            'unk_token_id': 1,
            'bos_token_id': 2,
            'eos_token_id': 3,
            'mask_token_id': 4
        }
    
    def encode_text(self, text: str) -> List[int]:
        """Encode text to token ids"""
        tokens = [self.tokenizer['bos_token_id']]
        
        for char in text:
            if char in self.tokenizer['char_to_idx']:
                tokens.append(self.tokenizer['char_to_idx'][char])
            else:
                tokens.append(self.tokenizer['unk_token_id'])
        
        tokens.append(self.tokenizer['eos_token_id'])
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([self.tokenizer['pad_token_id']] * (self.max_length - len(tokens)))
        
        return tokens
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.encode_text(text)
        
        # For language modeling, input and target are the same sequence shifted by 1
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': target_ids,
            'attention_mask': torch.ones_like(input_ids)
        }

class ImprovedLLMTrainer:
    """Enhanced trainer with better optimization techniques"""
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.scaler = GradScaler() if device == 'cuda' else None
        
        # Enhanced training parameters
        self.learning_rate = 2e-4
        self.weight_decay = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8
        
        # Learning rate scheduling
        self.warmup_steps = 1000
        self.total_steps = 10000
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        
        # Cosine annealing scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.total_steps,
            pct_start=self.warmup_steps / self.total_steps,
            div_factor=25,
            final_div_factor=10000
        )
        
        # Logging
        self.logger = self._setup_logging()
        self.writer = SummaryWriter('runs/enhanced_training')
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def get_lr_scale(self, step):
        """Get learning rate scale factor for warmup and decay"""
        if step < self.warmup_steps:
            return step / self.warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
    
    def compute_loss(self, batch):
        """Compute loss with better numerical stability"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with autocast():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.last_hidden_state
                
                # Compute cross-entropy loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.tokenizer['pad_token_id'],
                    label_smoothing=0.1  # Label smoothing for better generalization
                )
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.last_hidden_state
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.tokenizer['pad_token_id'],
                label_smoothing=0.1
            )
        
        return loss, logits
    
    def train_step(self, batch, step):
        """Single training step with improved optimization"""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, logits = self.compute_loss(batch)
        
        # Backward pass with gradient scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self, eval_dataloader):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                loss, _ = self.compute_loss(batch)
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_samples += batch['input_ids'].size(0)
        
        return total_loss / total_samples
    
    def generate_sample(self, prompt: str, max_length: int = 100):
        """Generate a sample response"""
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer['char_to_idx'].get(c, self.tokenizer['unk_token_id']) 
                                 for c in prompt], dtype=torch.long).unsqueeze(0).to(self.device)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                if generated.size(1) >= 512:  # Max sequence length
                    break
                    
                outputs = self.model(generated)
                logits = outputs.last_hidden_state[:, -1, :]
                
                # Sample from top-k with temperature
                temperature = 0.8
                top_k = 50
                
                logits = logits / temperature
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(-1, next_token_idx)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop at EOS token
                if next_token.item() == self.tokenizer['eos_token_id']:
                    break
        
        # Decode generated sequence
        generated_text = ''.join([self.tokenizer['idx_to_char'].get(idx.item(), '<UNK>') 
                                 for idx in generated[0] if idx.item() not in [
                                     self.tokenizer['pad_token_id'], 
                                     self.tokenizer['bos_token_id'], 
                                     self.tokenizer['eos_token_id']
                                 ]])
        
        return generated_text
    
    def train(self, train_dataloader, eval_dataloader, num_epochs: int = 5):
        """Main training loop with improvements"""
        self.logger.info(f"Starting enhanced training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            for batch in train_dataloader:
                loss = self.train_step(batch, step)
                epoch_loss += loss
                num_batches += 1
                step += 1
                
                # Logging
                if step % 100 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.logger.info(f"Step {step}: Loss = {loss:.4f}, LR = {current_lr:.6f}")
                    self.writer.add_scalar('Training/Loss', loss, step)
                    self.writer.add_scalar('Training/LearningRate', current_lr, step)
                
                # Evaluation
                if step % 500 == 0:
                    eval_loss = self.evaluate(eval_dataloader)
                    self.logger.info(f"Evaluation Loss: {eval_loss:.4f}")
                    self.writer.add_scalar('Evaluation/Loss', eval_loss, step)
                    
                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_checkpoint(step, eval_loss, is_best=True)
                    
                    # Generate sample
                    sample = self.generate_sample("Habari")
                    self.logger.info(f"Sample generation: {sample}")
            
            avg_epoch_loss = epoch_loss / num_batches
            self.logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(step, avg_epoch_loss, is_best=False)
    
    def save_checkpoint(self, step: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'tokenizer': self.tokenizer
        }
        
        checkpoint_path = f'checkpoints/checkpoint_step_{step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = 'checkpoints/best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at step {step} with loss {loss:.4f}")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

def main():
    """Main training function"""
    print("🚀 Starting Enhanced Trilingual LLM Training")
    print("=" * 50)
    
    # Create enhanced dataset
    print("📚 Creating enhanced dataset...")
    enhanced_texts = create_enhanced_dataset()
    print(f"Dataset size: {len(enhanced_texts)} examples")
    
    # Create dataset
    dataset = EnhancedTextDataset(enhanced_texts, max_length=256)
    print(f"Vocabulary size: {dataset.tokenizer['vocab_size']}")
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Create model (simplified for laptop training)
    from models.simple_llm import SimpleLLM, SimpleConfig
    
    config = SimpleConfig(
        vocab_size=dataset.tokenizer['vocab_size'],
        hidden_size=256,  # Smaller for faster training
        num_layers=4,     # Fewer layers
        num_heads=4,      # Fewer attention heads
        max_length=256
    )
    
    model = SimpleLLM(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ImprovedLLMTrainer(model, dataset.tokenizer)
    
    # Start training
    print("\n🎯 Starting training...")
    trainer.train(train_dataloader, eval_dataloader, num_epochs=10)
    
    print("\n✅ Training completed!")
    print("📁 Checkpoints saved in 'checkpoints/' directory")
    print("📊 Logs available in 'training.log' and TensorBoard 'runs/' directory")

if __name__ == "__main__":
    main()
