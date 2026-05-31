---
title: "Notes from AWS Unicorn Day Seoul 2026"
date: 2026-03-19
categories: [events]
tags: [aws, text2sql, ontology, cdk]
translation_key: aws-unicorn-day-seoul-2026
---

Through several company examples and sessions, I was able to see what kinds of synergy can emerge when development with AI is combined with AWS.

In this post, I want to summarize the parts that stood out to me from the sessions I attended, along with key terms and concepts that came up during the talks. Since this is a reconstruction based on brief notes I took on site, some wording or details may differ slightly from the actual presentations.

---

## From Implementing Text2SQL to Reducing the Data Team's Workload: Practical Operations Tips

> Son Hoeyeon, Solutions Architect; Park Seoyoung, Solutions Architect (AWS)
> 
> As data-driven decision-making becomes increasingly important in business settings, it is still difficult for people who do not know SQL to access data directly. This session covers how to quickly implement Text2SQL in a startup environment using LLMs, prompt engineering, and RAG, and shares practical know-how for improving accuracy in real services.

### Text2SQL implementation and operations tips

When operating Text2SQL, it seemed important to design constraints and operational metrics together so that users can reliably get the results they want, rather than focusing only on the ability to convert natural language into SQL.

1. **Constraints must be clearly defined for multi-turn queries.**  
   Users often continue their questions across multiple turns, so it is important to clearly limit how much previous conversational context should be reflected and which tables and columns can be used.

2. **Few-shot examples and schema pruning work well together.**  
   Providing suitable examples to the LLM and excluding schema information that is not relevant to the query can reduce noise. As a result, you can expect more accurate and consistent SQL generation.

3. **A/B testing should be performed based on user feedback.**  
   To check whether generated SQL matches the user's actual intent, it is necessary to collect user feedback and experimentally compare the effects of changes to prompts or model configuration.

4. **Dynamic model selection can be considered.**  
   Instead of using the same model for every query, this approach selects an appropriate model based on query difficulty, cost, and latency requirements.

For observability metrics to understand performance, you can use **response time**, **average number of turns per session**, **SQL generation success rate**, **user feedback results**, and similar indicators. In an AWS environment, Amazon Bedrock and Amazon CloudWatch can be used together to observe model calls and application operations metrics.

### Terms

- **Schema pruning**  
  A method of selecting only the tables, columns, and relationships from the full database schema that are highly relevant to the current question and passing them to the LLM. Reducing unnecessary schema information can lower the chance that the model references the wrong table or generates incorrect joins.

- **Dynamic model selection**  
  A strategy for dynamically choosing which model to use based on request complexity, cost, latency, and accuracy requirements. For example, simple queries can be handled by a cheaper and faster model, while complex analytical queries can be handled by a more capable model.

---

## Building an Ontology with Our Service's Data

> Park Jinwoo, Solutions Architect (AWS)
> 
> This session explores ontology, a topic that many customers have recently been thinking about, and explains how to build ontology on AWS. It covers how to graph data using AWS Agentic AI services, Neptune, RDB, and analytics services, and how to add existing structured and unstructured data to an ontology. It also presents ways to use agents to remove data silos and apply the results in service, planning, and marketing.

What is a good approach if you want to apply LLMs while effectively making use of existing database assets? In this session, one answer was **ontology** and **graph-based data usage**.

An ontology is a way to explicitly define the concepts, components, relationships, conditions, and entities used within a specific domain. It is also closely connected to knowledge graphs and the Semantic Web.

The core use cases are integrating scattered data, inferring implicit information based on relationships between data, and better understanding user intent. For example, if you manage a logistics system, you could implement a digital twin to experiment with various scenarios and use the results to suggest new processes.

However, there are practical barriers to data integration. It is easy to think, "Why not put all the data in one place and ask AI about it?" But in reality, "putting all the data in one place" itself is very difficult.

The representative reasons are as follows.

1. **Tacit knowledge exists.**  
   Knowledge such as personal experience, know-how, and intuition is difficult to turn into data because it is not clearly documented or represented in systems.

2. **Data silos exist.**  
   Different teams may use different formats, storage systems, and terminology. Even the same word, such as "user," may have different meanings across teams.

Therefore, ontology should not be built by indiscriminately integrating all data. Instead, it is more suitable to build it by selecting the necessary data first, centered on a **business objective**. Final decisions should also consider the overall context rather than looking only at individual pieces of data.

You can organize what data to include in an ontology through questions like these:

1. What problem are we trying to solve?
2. Does the data needed for that purpose actually exist?
3. How is the data collected and loaded?
4. How is the data currently being used?

One approach introduced for implementing ontology in an AWS environment was to use the open-source SDK **Strands Agents** together with **Amazon Neptune**, AWS's graph database. This approach can be extended into a pattern where an agent receives natural language queries, converts them into graph database queries, explores graph relationships, and then explains the results back to the user.

Amazon Redshift's Zero-ETL integration also helps connect data from operational databases to analytics environments more easily, allowing fresher data to be used for analytics, AI/ML, and reporting. However, Zero-ETL does not eliminate every data transformation process, nor does it make a database immediately and perfectly understandable to an LLM. Data modeling, permissions, quality management, and business terminology still need to be designed separately.

It is also worth carefully deciding whether a graph DB is truly necessary when implementing an enterprise ontology. For example, travel information connects many domains such as flights, accommodation, tourism, and restaurants, so a graph DB may seem suitable. On the other hand, in actual service implementation, the join structure of an existing RDB may be easier, faster, and more intuitive.

In the end, I felt that the important question is not "Should we use a graph DB?" but **whether the graph model provides enough value compared with the problem we are trying to solve and the complexity of the data relationships**.

### Terms

- **OWL (Web Ontology Language)**  
  A standard language used to express ontologies on the web. It can define classes, properties, relationships, constraints, and other elements in a machine-understandable form, and is used in Semantic Web and knowledge graph implementations.

- **Semantic Web**  
  A concept that aims to assign meaning and relationships to information on the web so that machines, not only humans, can understand and process the meaning of data. Ontology, RDF, and OWL are often mentioned together in this context.

- **Digital Twin**  
  A model that replicates a real-world system, device, process, or space in a digital environment. It can be used to monitor current state based on operational data or simulate what outcomes may occur under certain conditions.

- **Data silo**  
  A state in which data is separated by department, system, or service and is not connected across boundaries. When data silos are severe, information about the same customer or product can be scattered across multiple systems, making it difficult to understand the full context.

- **Strands Agents**  
  An open-source AI agent SDK released by AWS. It enables a model-driven approach to building AI agents and can integrate not only with AWS services such as Amazon Bedrock, but also with various external models and tools.

- **Zero-ETL**  
  An approach intended to reduce the burden of building separate ETL pipelines and make it easier to connect data from operational data sources to analytics systems. AWS provides Zero-ETL integrations between Amazon Redshift and several data sources, with the goal of using fresher data for analytics and AI/ML.

### References

- [Introducing Strands Agents, an Open Source AI Agents SDK](https://aws.amazon.com/ko/blogs/tech/introducing-strands-agents-an-open-source-ai-agents-sdk/)
- [Neptune Graph Analytics using Strands Agent](https://builder.aws.com/content/33Y7trPz5dvINmMpzlGRS5aQZ9A/neptune-graph-analytics-using-strands-agent)

---

## Building AWS Serverless OpenClaw with Vibe Coding

> Jung Dohyun, Principal Consultant (Roboco Co., Ltd.)
> 
> This session shares practical know-how from building the Serverless-OpenClaw project in just one day by using vibe coding to migrate OpenClaw, a recent open-source project that has drawn attention, to AWS serverless infrastructure. Based on the speaker's long experience as a software developer and technical trainer at AWS, he designed an architecture that combines Fargate, Lambda, API Gateway, and DynamoDB to maintain strong security while achieving operating costs of around $1 per month. The session covers the full process of implementing architecture design, security hardening, and cost optimization strategies through vibe coding, and introduces practical best practices such as TDD-based quality assurance, interview-based design, incremental implementation, and prompt strategies for effectively giving context to AI.

Personally, this was the session that left the strongest impression on me. I had been used to configuring deployment steps one by one myself, so it felt especially new to learn that by using the AWS CLI, a significant portion of deployment work can be delegated to an agent.

Of course, entrusting deployment to an agent does not mean that every process automatically becomes safe. If anything, you need to define constraints, validation pipelines, cost ceilings, and security standards even more carefully. So in this summary, I focused more on the prompt strategy and development approach used to implement the architecture than on the architecture itself.

First, you need to run an interview session to design the deployment architecture according to the nature of the project. In this session, cost optimization is set as the main goal, and requirements are made concrete based on an AWS CDK stack. During the interview, various trade-offs are compared, and the maximum monthly cost suitable for the project is fixed.

For example, I was willing to spend up to about $20 per month on a personal project, so I was recommended a Lightsail instance. I then used Docker to run the frontend, backend, and database all on that instance. Considering future scalability, it was cheaper and easier to manage than the Railway + Vercel combination I had used before, and I was also satisfied with its performance and speed.

Another point that stood out to me was the validation approach. Having a person manually verify everything on every deployment is inefficient, and Human in the Loop (HIL) can become a bottleneck. Instead, it may be more realistic to have AI review things once more, receive the result as a report, and let a person perform the final check.

The session introduced an approach that forces every commit to pass the following validation pipeline.

1. **TDD-based validation**  
   The implementation is modified until all test cases pass.

2. **Pre-commit hook**  
   Before committing, ESLint, Vitest, type checks, and similar validations are run.

3. **Pre-push hook**  
   Before pushing to the remote repository, E2E tests and CDK Synth validation are performed.

4. **Validation of README constraints**  
   For example, it checks whether NAT Gateway usage is prohibited and whether the monthly cost ceiling is being respected.

5. **Cost and security checklist validation**  
   Skills or checklists such as `/cost` and `/security` are used to review cost and security requirements.

At first, applying this kind of validation pipeline directly may seem complicated. But if you clone the GitHub repository in the references and ask an agent, "I want to apply this validation pipeline to my project," you can guide it to write the automation code.

### Terms

- **CDK Synth**  
  The process of checking whether AWS CDK code is correctly converted into a CloudFormation template. It is used to validate that infrastructure definitions are correct before deployment.

### References

- <https://github.com/serithemage/serverless-openclaw>
