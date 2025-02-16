 **大语言模型（LLM）科普：解析 ChatGPT 及相关技术**  

## **1. 引言**  
- LLM 训练过程、思维方式

---

## **2. 预训练阶段**  
### **2.1 预训练数据**（00:01:00）  

FineWeb (pretraining dataset): https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1

- **数据来源**：互联网（书籍、论文、网站、代码库）  
- **数据筛选**：去重、去噪、过滤低质量内容  
- **数据局限**：缺乏最新信息，可能存在偏见  

### **2.2 Tokenization（分词处理）**（00:07:47）  

Tiktokenizer: https://tiktokenizer.vercel.app/

- **BPE（Byte Pair Encoding）**：基于子词的分词方式  
- **Token 数量 vs. 计算成本**：Token 多→计算量大  
- **Token 化影响**：模型依赖 Token 组合 → 影响拼写、数学计算  

### **2.3 神经网络 I/O**（00:14:27） 

Transformer Neural Net 3D visualizer: https://bbycroft.net/llm

- **输入**：Token 序列 → 经过嵌入层处理  
- **输出**：Token 概率分布 → 生成文本  
- **神经网络内部机制**（00:20:11） 

### **2.4 推理（Inference）**（00:26:01） 

- **推理过程**：基于训练数据生成最可能的文本  

- **GPT-2**（00:31:09）  
  - **架构**：基于 Transformer 的自回归模型  
  - **训练方法**：SFT（监督微调），预测下一个 Token  

- **Llama 3.1**（00:42:52）  
  - **In-context learning (ICL)** （00:57:00）  

---

## **3. 预训练 vs. 后训练**（00:59:23）  
- 预训练：学习通用语言模式  
- 后训练：通过微调提升任务适应性  

## **4. 后训练阶段**  
### **4.1 对话数据**（01:01:06）  
- **人类反馈**：提供数据训练模型，优化回答质量  
- **数据来源**：对话、人工标注文本  


### **4.2 幻觉（Hallucination）**（01:20:32）

### **4.3 LLM 的自我认知**（01:41:46）  
- **是否“理解”自身？**  
  - LLM 没有真正的自我意识  
  - 只能基于训练数据模拟反思  

### **4.4 Token 依赖性**（01:46:56）  
- **模型“思考”依赖 Token**  
- **把所有计算集中在一个token中准确度会下降**

### **4.5 工具使用**（01:56:01） 

### **4.6 其他短板**（02:01:11）  
- **LLM 依赖 Token 进行计算，而非逐字处理** → 拼写、分词、特殊符号处理能力受限 
- **Model Can’t Count**
---
## **5. 监督微调 vs. 强化学习**（02:07:28）  
- 监督学习：基于标注数据学习固定映射  
- 强化学习：通过试错优化策略  

### 

## **6. 强化学习（RL）**  
### **6.1 强化学习机制**（02:14:42）  
- **PPO（近端策略优化）**：稳定强化学习训练  
- **奖励模型（Reward Model）**：学习人类偏好，提高模型生成质量  

### **6.2 具体案例**  
#### **DeepSeek-R1（02:27:47）**  
- **结合 SFT（监督微调）+ RLHF（强化学习）**  
- **目标**：提升 AI 推理能力  

#### **AlphaGo（02:42:07）**  
- **深度强化学习（Deep RL）+ 蒙特卡洛树搜索（MCTS）**  
- **策略**：自我对弈不断优化  


### **6.3 RLHF 的局限性（02:48:26）** 

Reinforcement learning from human feedback

- **对抗样本**：adversarial example
- **比起生成，人类更擅长辨别**

---

## **7. 未来展望**  
### **7.1 未来趋势**（03:09:39）  
- **多模态**
- **AI agent**
- ...

### **7.2 如何追踪 LLM 发展**（03:15:15）  
- LM Arena for model rankings: https://lmarena.ai/
- AI News Newsletter: https://buttondown.com/ainews
- X


### **7.3 LLM 的获取方式**（03:18:34）  
- LMStudio for local inference https://lmstudio.ai/


## **8 链接**

### 原视频

- https://youtu.be/7xTGNNLPyMI?si=IGZjWwhuMsyVmLDG


### 导图

The visualization UI I was using in the video:
- https://excalidraw.com/

The specific file of Excalidraw we built up:
- https://drive.google.com/file/d/1EZh5hNDzxMMy05uLhVryk061QYQGTxiN/view?usp=sharing.

### 工具
 
FineWeb (pretraining dataset):
- https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1

Tiktokenizer:
- https://tiktokenizer.vercel.app/

Transformer Neural Net 3D visualizer:
- https://bbycroft.net/llm

llm.c Let's Reproduce GPT-2：
- https://github.com/karpathy/llm.c/discussions/677

Llama 3 paper from Meta:
- https://arxiv.org/abs/2407.21783

Hyperbolic, for inference of base model:
- https://app.hyperbolic.xyz/

InstructGPT paper on SFT:
- https://arxiv.org/abs/2203.02155

HuggingFace inference playground:
- https://huggingface.co/spaces/huggingface/inference-playground

DeepSeek-R1 paper:
- https://arxiv.org/abs/2501.12948

TogetherAI Playground for open model inference:
- https://api.together.xyz/playground

LM Arena for model rankings:
- https://lmarena.ai/

AI News Newsletter:
- https://buttondown.com/ainews

LMStudio for local inference:
- https://lmstudio.ai/
