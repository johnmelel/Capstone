"""
Basic RL training for query rewriting using REINFORCE
Trains GPT-2 small to rewrite queries for better vector search
"""
import os
import json
import torch
import random
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from pymilvus import MilvusClient
import vertexai
from vertexai.language_models import TextEmbeddingModel
import config

# Load environment
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

class QueryRewriter:
    """Simple RL agent to rewrite queries"""
    
    def __init__(self):
        print(f"Loading model: {config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.optimizer = AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def rewrite_query(self, query):
        """Generate rewritten query"""
        prompt = f"Rewrite this medical query to be more specific: {query}\nRewritten:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with sampling (needed for RL)
        outputs = self.model.generate(
            **inputs,
            max_length=config.MAX_QUERY_LENGTH + len(inputs["input_ids"][0]),
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Decode rewritten query
        full_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        rewritten = full_text.split("Rewritten:")[-1].strip()
        
        return rewritten, outputs.sequences, outputs.scores
    
    def compute_log_probs(self, sequences, scores):
        """Compute log probabilities for generated sequence"""
        log_probs = []
        for i, score in enumerate(scores):
            # Get probability of selected token
            token_id = sequences[0][len(sequences[0]) - len(scores) + i]
            log_prob = torch.log_softmax(score[0], dim=-1)[token_id]
            log_probs.append(log_prob)
        return torch.stack(log_probs).sum()
    
    def update(self, log_prob, reward):
        """REINFORCE update: maximize log_prob * reward"""
        loss = -log_prob * reward  # Negative because we minimize loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path):
        """Save model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")


class RewardCalculator:
    """Calculate reward based on vector search results"""
    
    def __init__(self):
        print("Connecting to Milvus...")
        self.client = MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)
        
        print("Initializing embeddings...")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(config.SERVICE_ACCOUNT)
        vertexai.init(project=config.GCP_PROJECT, location=config.GCP_LOCATION)
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        print("Reward calculator ready")
    
    def get_embedding(self, text):
        """Get embedding for text"""
        embeddings = self.embedding_model.get_embeddings([text])
        return embeddings[0].values
    
    def search(self, query):
        """Search Milvus with query"""
        embedding = self.get_embedding(query)
        
        results = self.client.search(
            collection_name=config.COLLECTION_NAME,
            data=[embedding],
            limit=config.TOP_K_RESULTS,
            output_fields=["text"]
        )
        
        return results[0] if results else []
    
    def calculate_reward(self, original_query, rewritten_query, source_text):
        """
        Simple reward:
        +1 if retrieved chunks contain keywords from original query
        -0.01 per token (length penalty)
        """
        # Search with rewritten query
        results = self.search(rewritten_query)
        
        if not results or len(results) == 0:
            return -1.0  # No results = bad
        
        # Check if source text keywords appear in retrieved chunks
        source_keywords = set(source_text.lower().split()[:20])  # First 20 words
        
        retrieved_texts = [r["entity"]["text"].lower() for r in results]
        retrieved_words = set(" ".join(retrieved_texts).split())
        
        # Keyword overlap
        overlap = len(source_keywords & retrieved_words) / len(source_keywords)
        
        # Length penalty
        length_penalty = len(rewritten_query.split()) * config.REWARD_LENGTH_PENALTY
        
        reward = overlap - length_penalty
        return reward


def train(agent, reward_calc, questions, num_episodes=100):
    """Train agent with REINFORCE"""
    
    print(f"\nTraining for {num_episodes} episodes...")
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Sample a question
        q_data = random.choice(questions)
        original_query = q_data["question"]
        source_text = q_data["source_text"]
        
        # Generate rewritten query
        rewritten_query, sequences, scores = agent.rewrite_query(original_query)
        
        # Calculate reward
        reward = reward_calc.calculate_reward(original_query, rewritten_query, source_text)
        
        # Compute log probability of generated sequence
        log_prob = agent.compute_log_probs(sequences, scores)
        
        # Update model
        loss = agent.update(log_prob, reward)
        
        episode_rewards.append(reward)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.3f}")
            print(f"  Original: {original_query[:60]}...")
            print(f"  Rewritten: {rewritten_query[:60]}...")
    
    # Save training log
    log_file = config.LOGS_DIR / "training_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            "episodes": num_episodes,
            "rewards": episode_rewards,
            "final_avg_reward": sum(episode_rewards[-10:]) / 10
        }, f, indent=2)
    
    print(f"\nTraining log saved to {log_file}")
    return episode_rewards


def main():
    """Main training loop"""
    
    # Load questions
    if not config.QUESTIONS_FILE.exists():
        print(f"Error: {config.QUESTIONS_FILE} not found")
        print("Run 1_generate_questions.py first")
        return
    
    with open(config.QUESTIONS_FILE, 'r') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    # Initialize
    agent = QueryRewriter()
    reward_calc = RewardCalculator()
    
    # Train
    rewards = train(agent, reward_calc, questions, config.NUM_EPISODES)
    
    # Save model
    model_path = config.MODEL_DIR / "query_rewriter"
    agent.save(model_path)
    
    # Show results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Average reward (first 10): {sum(rewards[:10])/10:.3f}")
    print(f"Average reward (last 10): {sum(rewards[-10:])/10:.3f}")
    print(f"Improvement: {(sum(rewards[-10:])/10 - sum(rewards[:10])/10):.3f}")
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
