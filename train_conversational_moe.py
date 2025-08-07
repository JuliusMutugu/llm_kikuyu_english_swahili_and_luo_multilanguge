"""
Quick training script for Conversational MoE Model
Simple, efficient training for laptop deployment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import datetime
from conversational_moe_model import ConversationalMoESystem

class ConversationDataset(Dataset):
    """Simple conversation dataset"""
    def __init__(self, conversations, tokenizer_func, max_length=128):
        self.conversations = conversations
        self.tokenizer_func = tokenizer_func
        self.max_length = max_length
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        # Create input-output pairs
        input_text = conv['input']
        target_text = conv['output']
        
        input_tokens = self.tokenizer_func(input_text, self.max_length)
        target_tokens = self.tokenizer_func(target_text, self.max_length)
        
        return {
            'input_ids': torch.tensor(input_tokens, dtype=torch.long),
            'target_ids': torch.tensor(target_tokens, dtype=torch.long)
        }

def create_training_data():
    """Create enhanced multilingual training data for conversations"""
    conversations = [
        # English conversations
        {"input": "Hello, how are you?", "output": "I'm doing well, thank you! How can I help you today?"},
        {"input": "What can you do?", "output": "I can help with conversations in multiple languages including English, Kiswahili, Kikuyu, and Luo."},
        {"input": "Tell me a story", "output": "Once upon a time, there was a helpful AI that loved to chat with people from all over the world."},
        {"input": "How do you work?", "output": "I use a mixture of experts model to understand and respond in different languages."},
        {"input": "What languages do you speak?", "output": "I can communicate in English, Kiswahili, Kikuyu, and Luo languages."},
        
        # Kiswahili conversations
        {"input": "Habari yako?", "output": "Nzuri sana, asante. Unaweza kunisaidia nini leo?"},
        {"input": "Unaongea lugha gani?", "output": "Naongea Kiingereza, Kiswahili, Kikuyu, na Kiluo."},
        {"input": "Unaweza kunisaidia?", "output": "Ndiyo, nina furaha kukusaidia. Je, unahitaji msaada gani?"},
        {"input": "Nakupenda", "output": "Asante sana! Mimi pia nakupenda. Tuendelee mazungumzo."},
        {"input": "Unafahamu nini?", "output": "Nafahamu mengi kuhusu lugha mbalimbali na mazungumzo."},
        
        # Kikuyu conversations  
        {"input": "Wĩ atĩa?", "output": "Ndĩ mwega mũno, nĩngũkwenda. Ũngĩndeithagia atĩa ũmũthĩ?"},
        {"input": "Nĩngũkwenda mũno", "output": "Nĩngũkenagia mũno! Nĩngũkwenda o nawe."},
        {"input": "Ũraaria ciũgano irĩkũ?", "output": "Ndaaria Gĩkũyũ, Kiswahili, Kiingereza, na Kiluo."},
        {"input": "Ũngĩnyonia ũndũ?", "output": "Ĩĩ, nĩngũgũteithia. Ũrenda atĩa?"},
        {"input": "Njĩra cia gũthoma", "output": "Gũthoma nĩ wega mũno. Rĩrĩa ũrathoma, meciiria maku nĩmagĩa hinya."},
        
        # Luo conversations
        {"input": "Inadi?", "output": "Adhi maber ahinya. Inyalo konya nadi?"},
        {"input": "Aheri miwuoro", "output": "Ahinya miwuoro bende! Ere gima dimito?"},
        {"input": "Iwacho dhok mage?", "output": "Awacho Luo, Kiswahili, English kod Kikuyu."},
        {"input": "Inyalo konya?", "output": "Ee, anyalo konyoni. En gima imito mondo atimni?"},
        {"input": "Nyis ane gimoro", "output": "Gimoro maber: dhano duto nyalo puonjore gik manyien pile ka pile."},
        
        # Mixed language conversations
        {"input": "Hello, unaweza kuongea Kiswahili?", "output": "Yes, naongea Kiswahili pia! Tuongee kwa lugha yoyote unayotaka."},
        {"input": "Habari, ũraria Gĩkũyũ?", "output": "Ĩĩ, ndaaria Gĩkũyũ! Wĩ mwega?"},
        {"input": "Inadi, can you speak English?", "output": "Yes, I can speak English. Adhi maber bende!"},
        
        # Practical conversations
        {"input": "I need help with homework", "output": "I'd be happy to help! What subject are you working on?"},
        {"input": "Nahitaji msaada wa kazi", "output": "Ninafuraha kukusaidia! Ni kazi gani unayofanya?"},
        {"input": "How do I learn a new language?", "output": "Start with basic words, practice daily, and don't be afraid to make mistakes!"},
        {"input": "Njĩra ya gũthoma rũthiomi rũrũ", "output": "Ambĩrĩria na ciugo nyingĩ, wĩhũũre o mũthenya, na ndũketigĩre kũhĩtia!"},
        
        # Emotional conversations
        {"input": "I'm feeling sad today", "output": "I'm sorry to hear that. Would you like to talk about what's making you feel sad?"},
        {"input": "Nimo na furaha sana", "output": "Hilo ni jambo jema sana! Ni nini kinachokufanya ufurahi hivyo?"},
        {"input": "Ndĩ na kĩeha", "output": "Nĩnjiguaga ũũru waku. Nĩ kĩĩ kĩragũtĩĩa kĩeha?"},
        
        # Educational conversations
        {"input": "Tell me about science", "output": "Science helps us understand the world around us through observation and experiments."},
        {"input": "Niambie kuhusu sayansi", "output": "Sayansi hutusaidia kuelewa ulimwengu kupitia uchunguzi na majaribio."},
        {"input": "What is mathematics?", "output": "Mathematics is the language of numbers, patterns, and logical reasoning."},
        
        # Cultural conversations
        {"input": "Tell me about African culture", "output": "African cultures are rich and diverse, with beautiful traditions, languages, and customs."},
        {"input": "Niambie kuhusu utamaduni wa Afrika", "output": "Utamaduni wa Afrika ni tajiri na wa kipekee, una desturi nzuri, lugha, na mila."},
        {"input": "Njĩra cia Agĩkũyũ", "output": "Agĩkũyũ nĩ andũ a ũtũũro mwega, marĩ na mĩtugo mĩega na ũũgĩ mũingĩ."},
    ]
    
    # Add more training examples by creating variations
    extended_conversations = []
    for conv in conversations:
        extended_conversations.append(conv)
        
        # Create reverse conversations for better bidirectional learning
        reverse_conv = {
            "input": conv["output"],
            "output": "That's interesting! Tell me more about that."
        }
        extended_conversations.append(reverse_conv)
    
    print(f"Created {len(extended_conversations)} training conversations")
    return extended_conversations

def train_conversational_moe(epochs=5, save_path="trained_conversational_moe.pt"):
    """Train the conversational MoE model"""
    print("🚀 Starting Conversational MoE Training...")
    
    # Initialize system
    moe_system = ConversationalMoESystem()
    device = moe_system.device
    model = moe_system.model
    
    # Create training data
    conversations = create_training_data()
    dataset = ConversationDataset(conversations, moe_system.tokenize)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Small batch size for laptop
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    # Training loop
    model.train()
    total_loss = 0
    step = 0
    
    print(f"📚 Training on {len(conversations)} conversations for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs['logits']
            
            # Calculate loss
            # Shift targets for autoregressive training
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Add expert balancing loss if available
            routing_info = outputs.get('routing_info', {})
            if 'expert_usage' in routing_info:
                expert_usage = routing_info['expert_usage']
                # Encourage balanced expert usage
                balance_loss = torch.var(expert_usage) * 0.01
                loss += balance_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            num_batches += 1
            step += 1
            
            if step % 10 == 0:
                avg_loss = total_loss / step
                print(f"Step {step}: Loss = {loss.item():.4f}, Avg Loss = {avg_loss:.4f}")
        
        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        scheduler.step()
        
        print(f"✅ Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        # Test generation after each epoch
        if (epoch + 1) % 2 == 0:
            model.eval()
            test_prompt = "Hello, how are you?"
            result = moe_system.generate_response(test_prompt, max_length=30)
            print(f"🧪 Test Response: '{result['response']}'")
            model.train()
    
    # Final evaluation
    model.eval()
    print("\n🧪 Final Testing:")
    
    test_prompts = [
        "Hello friend",
        "Habari yako?", 
        "Wĩ atĩa?",
        "Inadi?"
    ]
    
    for prompt in test_prompts:
        result = moe_system.generate_response(prompt, max_length=40)
        print(f"   Input: '{prompt}' → Output: '{result['response']}'")
    
    # Save the trained model
    moe_system.save_model(save_path)
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Training Complete!")
    print(f"   Final Average Loss: {total_loss / step:.4f}")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Model saved to: {save_path}")
    
    return moe_system

if __name__ == "__main__":
    # Train the model
    trained_system = train_conversational_moe(epochs=8, save_path="conversational_moe_trained.pt")
    print("✅ Conversational MoE training completed!")
