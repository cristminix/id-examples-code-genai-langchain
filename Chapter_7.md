# Software Development and Data Analysis Agents

This chapter explores how natural language—our everyday English or whatever language you prefer to interact in with an LLM—has emerged as a powerful interface for programming, a paradigm shift that, when taken to its extreme, is called _vibe coding_. Instead of learning acquiring new programming languages or frameworks, developers can now articulate their intent in natural language, leaving it to advanced LLMs and frameworks such as LangChain to translate these ideas into robust, production-ready code. Moreover, while traditional programming languages remain essential for production systems, LLMs are creating new workflows that complement existing practices and potentially increase accessibility This evolution represents a significant shift from earlier attempts at code generation and automation.

We'll specifically discuss LLMs' place in software development and the state of the art of performance, models, and applications. We'll see how to use LLM chains and agents to help in code generation and data analysis, training ML models, and extracting predictions. We'll cover writing code with LLMs, giving examples with different models be it on Google's generative AI services, Hugging Face, or Anthropic. After this, we'll move on to more advanced approaches with agents and RAG for documentation or a code repository.

We'll also be applying LLM agents to data science: we'll first train a model on a dataset, then we'll analyze and visualize a dataset. Whether you're a developer, a data scientist, or a technical decision-maker, this chapter will equip you with a clear understanding of how LLMs are reshaping software development and data analysis while maintaining the essential role of conventional programming languages.

The following topics will be covered in this chapter:

- LLMs in software development
- Writing code with LLMs
- Applying LLM agents for data science

## LLMs in software development

The relationship between natural language and programming is undergoing a significant transformation. Traditional programming languages remain essential in software development—C++ and Rust for performance-critical applications, Java and C# for enterprise systems, and Python for rapid development, data analysis, and ML workflows. However, natural language, particularly English, now serves as a powerful interface to streamline software development and data science tasks, complementing rather than replacing these specialized programming tools.

Advanced AI assistants let you build software by simply staying "in the vibe" of what you want, without ever writing or even picturing a line of code. This style of development, known as vibe coding, was popularized by Andrej Karpathy in early 2025. Instead of framing tasks in programming terms or wrestling with syntax, you describe desired behaviors, user flows or outcomes in plain conversation. The model then orchestrates data structures, logic and integration behind the scenes. With vibe coding you don't debug—you re-vibe. This means, you iterate by restating or refining requirements in natural language, and let the assistant reshape the system. The result is a pure, intuitive design-first workflow that completely abstracts away all coding details.

Tools such as Cursor, Windsurf (formerly Codeium), OpenHands, and Amazon Q Developer have emerged to support this development approach, each offering different capabilities for AI-assisted coding. In practice, these interfaces are democratizing software creation while freeing experienced engineers from repetitive tasks. However, balancing speed with code quality and security remains critical, especially for production systems.

The software development landscape has long sought to make programming more accessible through various abstraction layers. Early efforts included fourth-generation languages that aimed to simplify syntax, allowing developers to express logic with fewer lines of code. This evolution continued with modern low-code platforms, which introduced visual programming with pre-built components to democratize application development beyond traditional coding experts. The latest and perhaps most transformative evolution features natural language programming through LLMs, which interpret human intentions expressed in plain language and translate them into functional code.

What makes this current evolution particularly distinctive is its fundamental departure from previous approaches. Rather than creating new artificial languages for humans to learn, we're adapting intelligent tools to understand natural human communication, significantly lowering the barrier to entry. Unlike traditional low-code platforms that often result in proprietary implementations, natural language programming generates standard code without vendor lock-in, preserving developer freedom and compatibility with existing ecosystems. Perhaps most importantly, this approach offers unprecedented flexibility across the spectrum, from simple tasks to complex applications, serving both novices seeking quick solutions and experienced developers looking to accelerate their workflow.

### The future of development

Analysts at International Data Corporation (IDC) project that, by 2028, natural language will be used to create 70% of new digital solutions (IDC FutureScape, _Worldwide Developer and DevOps 2025 Predictions_). However, this doesn't mean traditional programming will disappear; rather, it's evolving into a two-tier system where natural language serves as a high-level interface while traditional programming languages handle precise implementation details.

However, this evolution does not spell the end for traditional programming languages. While natural language can streamline the design phase and accelerate prototyping, the precision and determinism of languages like Python remain essential for building reliable, production-ready systems. In other words, rather than replacing code entirely, English (or Mandarin, or whichever natural language best suits our cognitive process) is augmenting it—acting as a high-level layer that bridges human intent with executable logic.

For software developers, data scientists, and technical decision-makers, this shift means embracing a hybrid workflow where natural language directives, powered by LLMs and frameworks such as LangChain, coexist with conventional code. This integrated approach paves the way for faster innovation, personalized software solutions, and, ultimately, a more accessible development process.

### Implementation considerations

For production environments, the current evolution manifests in several ways that are transforming how development teams operate. Natural language interfaces enable faster prototyping and reduce time spent on boilerplate code, while traditional programming remains essential for the optimization and implementation of complex features. However, recent independent research shows significant limitations in current AI coding capabilities.

The 2025 OpenAI _SWE-Lancer_ benchmark study found that even the top-performing model completed only 26.2% of individual engineering tasks drawn from real-world freelance projects. The research identified specific challenges including surface-level problem-solving, limited context understanding across multiple files, inadequate testing, and poor edge case handling.

Despite these limitations, many organizations report productivity gains when using AI coding assistants in targeted ways. The most effective approach appears to be collaboration—using AI to accelerate routine tasks while applying human expertise to areas where AI still struggles, such as architectural decisions, comprehensive testing, and understanding business requirements in context. As the technology matures, the successful integration of natural language and traditional programming will likely depend on clearly defining where each excels rather than assuming AI can autonomously handle complex software engineering challenges.

Code maintenance has evolved through AI-assisted approaches where developers use natural language to understand and modify codebases. While GitHub reports Copilot users completed specific coding tasks 55% faster in controlled experiments, independent field studies show more modest productivity gains ranging from 4–22%, depending on context and measurement approach. Similarly, Salesforce reports their internal CodeGenie tool contributes to productivity improvements, including automating aspects of code review and security scanning. Beyond raw speed improvements, research consistently shows AI coding assistants reduce developer cognitive load and improve satisfaction, particularly for repetitive tasks. However, studies also highlight important limitations: generated code often requires significant human verification and rework, with some independent research reporting higher bug rates in AI-assisted code. The evidence suggests these tools are valuable assistants that streamline development workflows while still requiring human expertise for quality and security assurance.

The field of code debugging has been enhanced as natural language queries help developers identify and resolve issues faster by explaining error messages, suggesting potential fixes, and providing context for unexpected behavior. AXA's deployment of "AXA Secure GPT," trained on internal policies and code repositories, has significantly reduced routine task turnaround times, allowing development teams to focus on more strategic work (AXA, _AXA offers secure Generative AI to employees_).

When it comes to understanding complex systems, developers can use LLMs to generate explanations and visualizations of intricate architectures, legacy codebases, or third-party dependencies, accelerating onboarding and system comprehension. For example, Salesforce's system landscape diagrams show how their LLM-integrated platforms connect across various services, though recent earnings reports indicate these AI initiatives have yet to significantly impact their financial results.

System architecture itself is evolving as applications increasingly need to be designed with natural language interfaces in mind, both for development and potential user interaction. BMW reported implementing a platform that uses generative AI to produce real-time insights via chat interfaces, reducing the time from data ingestion to actionable recommendations from days to minutes. However, this architectural transformation reflects a broader industry pattern where consulting firms have become major financial beneficiaries of the generative AI boom. Recent industry analysis shows that consulting giants such as Accenture are generating more revenue from generative AI services ($3.6 billion in annualized bookings) than most generative AI startups combined, raising important questions about value delivery and implementation effectiveness that organizations must consider when planning their AI architecture strategies.

For software developers, data scientists, and decision-makers, this integration means faster iteration, lower costs, and a smoother transition from idea to deployment. While LLMs help generate boilerplate code and automate routine tasks, human oversight remains critical for system architecture, security, and performance. As the case studies demonstrate, companies integrating natural language interfaces into development and operational pipelines are already realizing tangible business value while maintaining necessary human guidance.

### Evolution of code LLMs

The development of code-specialized LLMs has followed a rapid trajectory since their inception, progressing through three distinct phases that have transformed software development practices. The first _Foundation phase_ (2021 to early 2022) introduced the first viable code generation models that proved the concept was feasible. This was followed by the _Expansion phase_ (late 2022 to early 2023), which brought significant improvements in reasoning capabilities and contextual understanding. Most recently, the _Diversification phase_ (mid-2023 to 2024) has seen the emergence of both advanced commercial offerings and increasingly capable open-source alternatives.

This evolution has been characterized by parallel development tracks in both proprietary and open-source ecosystems. Initially, commercial models dominated the landscape, but open-source alternatives have gained substantial momentum more recently. Throughout this progression, several key milestones have marked transformative shifts in capabilities, opening new possibilities for AI-assisted development across different programming languages and tasks. The historical context of this evolution provides important insights for understanding implementation approaches with LangChain.

![Figure 7.1: Evolution of code LLMs (2021–2024)](Images/B32363_07_01.png)

_Figure 7.1_ illustrates the progression of code-specialized language models across commercial (upper track) and open-source (lower track) ecosystems. Key milestones are highlighted, showing the transition from early proof-of-concept models to increasingly specialized solutions. The timeline spans from early commercial models such as Codex to recent advancements such as Google's Gemini 2.5 Pro (March 2025) and specialized code models such as Mistral AI's Codestral series.

In recent years, we've witnessed an explosion of LLMs fine-tuned specifically tailored for coding—commonly known as code LLMs. These models are rapidly evolving, each with its own set of strengths and limitations, and are reshaping the software development landscape. They offer the promise of accelerating development workflows across a broad spectrum of software engineering tasks:

- **Code generation: **Transforming natural language requirements into code snippets or full functions. For instance, developers can generate boilerplate code or entire modules based on project specifications.
- **Test generation: **Creating unit tests from descriptions of expected behavior to improve code reliability.
- **Code documentation**: Automatically generating docstrings, comments, and technical documentation from existing code or specifications. This significantly reduces the documentation burden that often gets deprioritized in fast-paced development environments.
- **Code editing and refactoring: **Automatically suggesting improvements, fixing bugs, and restructuring code for maintainability.
- **Code translation: **Converting code between different programming languages or frameworks.
- **Debugging and automated program repair**: Identifying bugs within large codebases and generating patches to resolve issues. For example, tools such as SWE-agent, AutoCodeRover, and RepoUnderstander iteratively refine code by navigating repositories, analyzing abstract syntax trees, and applying targeted changes.

The landscape of code-specialized LLMs has grown increasingly diverse and complex. This evolution raises critical questions for developers implementing these models in production environments: Which model is most suitable for specific programming tasks? How do different models compare in terms of code quality, accuracy, and reasoning capabilities? What are the trade-offs between open-source and commercial options? This is where benchmarks become essential tools for evaluation and selection.

### Benchmarks for code LLMs

Objective benchmarks provide standardized methods to compare model performance across a variety of coding tasks, languages, and complexity levels. They help quantify capabilities that would otherwise remain subjective impressions, allowing for data-driven implementation decisions.

For LangChain developers specifically, understanding benchmark results offers several advantages:

- **Informed model selection:** Choosing the optimal model for specific use cases based on quantifiable performance metrics rather than marketing claims or incomplete testing
- **Appropriate tooling**: Designing LangChain pipelines that incorporate the right balance of model capabilities and augmentation techniques based on known model strengths and limitations
- **Cost-benefit analysis:** Evaluating whether premium commercial models justify their expense compared to free or self-hosted alternatives for particular applications
- **Performance expectations:** Setting realistic expectations about what different models can achieve when integrated into larger systems

Code-generating LLMs demonstrate varying capabilities across established benchmarks, with performance characteristics directly impacting their effectiveness in LangChain implementations. Recent evaluations of leading models, including OpenAI's GPT-4o (2024), Anthropic's Claude 3.5 Sonnet (2025), and open-source models such as Llama 3, show significant advancements in standard benchmarks. For instance, OpenAI's o1 achieves 92.4% pass@1 on HumanEval (_A Survey On Large Language Models For Code Generation_, 2025), while Claude 3 Opus reaches 84.9% on the same benchmark (_The Claude 3 Model Family: Opus, Sonnet, Haiku_, 2024). However, performance metrics reveal important distinctions between controlled benchmark environments and the complex requirements of production LangChain applications.

Standard benchmarks provide useful but limited insights into model capabilities for LangChain implementations:

- **HumanEval**This benchmark evaluates functional correctness through 164 Python programming problems. HumanEval primarily tests isolated function-level generation rather than the complex, multi-component systems typical in LangChain applications.
- **MBPP **(Mostly Basic Programming Problems): This contains approximately 974 entry-level Python tasks. These problems lack the dependencies and contextual complexity found in production environments.
- **ClassEval**: This newer benchmark tests class-level code generation, addressing some limitations of function-level testing. Recent research by Liu et al. (_Evaluating Large Language Models in Class-Level Code Generation_, 2024) shows performance degradation of 15–30% compared to function-level tasks, highlighting challenges in maintaining contextual dependencies across methods—a critical consideration for LangChain components that manage state.
- **SWE-bench**: More representative of real-world development, this benchmark evaluates models on bug-fixing tasks from actual GitHub repositories. Even top-performing models achieve only 40–65% success rates, as found by Jimenez et al. (_SWE-bench: Can Language Models Resolve Real-World GitHub Issues?_, 2023), demonstrating the significant gap between synthetic benchmarks and authentic coding challenges.

### LLM-based software engineering approaches

When implementing code-generating LLMs within LangChain frameworks, several key challenges emerge.

Repository-level problems that require understanding multiple files, dependencies, and context present significant challenges. Research using the ClassEval benchmark (Xueying Du and colleagues, _Evaluating Large Language Models in Class-Level Code Generation_, 2024) demonstrated that LLMs find class-level code generation "significantly more challenging than generating standalone functions," with performance consistently lower when managing dependencies between methods compared to function-level benchmarks such as HumanEval.

LLMs can be leveraged to understand repository-level code context despite the inherent challenges. The following implementation demonstrates a practical approach to analyzing multi-file Python codebases with LangChain, loading repository files as context for the model to consider when implementing new features. This pattern helps address the context limitations by directly providing a repository structure to the LLM:

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GitLoader
# Load repository context
repo_loader = GitLoader( clone_url="https://github.com/example/repo.git", branch="main", file_filter=lambda file_path: file_path.endswith(".py") ) documents = repo_loader.load()

# Create context-aware prompt
system_template = """You are an expert Python developer. Analyze the following repository files and implement the requested feature. Repository structure: {repo_context}"""
human_template = """Implement a function that: {feature_request}"""
prompt = ChatPromptTemplate.from_messages([ ("system", system_template), ("human", human_template) ])

# Create model with extended context window
model = ChatOpenAI(model="gpt-4o", temperature=0.2)
```

This implementation uses GPT-4o to generate code while considering the context of entire repositories by pulling in relevant Python files to understand dependencies. This approach addresses context limitations but requires careful document chunking and retrieval strategies for large codebases.

Generated code often appears superficially correct but contains subtle bugs or security vulnerabilities that evade initial detection. The Uplevel Data Labs study (_Can GenAI Actually Improve Developer Productivity?_) analyzing nearly 800 developers found a "significantly higher bug rate" in code produced by developers with access to AI coding assistants compared to those without. This is further supported by BlueOptima's comprehensive analysis in 2024 of over 218,000 developers (_Debunking GitHub's Claims: A Data-Driven Critique of Their Copilot Study_), which revealed that 88% of professionals needed to substantially rework AI-generated code before it was production-ready, often due to "aberrant coding patterns" that weren't immediately apparent.

Security researchers have identified a persistent risk where AI models inadvertently introduce security flaws by replicating insecure patterns from their training data, with these vulnerabilities frequently escaping detection during initial syntax and compilation checks (_Evaluating Large Language Models through Role-Guide and Self-Reflection: A Comparative Study_, 2024, and _HalluLens: LLM Hallucination Benchmark_, 2024). These findings emphasize the critical importance of thorough human review and testing of AI-generated code before production deployment.

The following example demonstrates how to create a specialized validation chain that systematically analyzes generated code for common issues, serving as a first line of defense against subtle bugs and vulnerabilities:

````python
from langchain.prompts import PromptTemplate
validation_template = """Analyze the following Python code for:
1. Potential security vulnerabilities
2. Logic errors
3. Performance issues
4. Edge case handling

Code to analyze:
```python
{generated_code}
Provide a detailed analysis with specific issues and recommended fixes. """

validation_prompt = PromptTemplate( input_variables=["generated_code"], template=validation_template )
validation_chain = validation_prompt | llm
````

This validation approach creates a specialized LLM-based code review step in the workflow, focusing on critical security and quality aspects.

Most successful implementations incorporate execution feedback, allowing models to iteratively improve their output based on compiler errors and runtime behavior. Research on Text-to-SQL systems by Boyan Li and colleagues (_The Dawn of Natural Language to SQL: Are We Fully Ready?_, 2024) demonstrates that incorporating feedback mechanisms significantly improves query generation accuracy, with systems that use execution results to refine their outputs and consistently outperform those without such capabilities.

When deploying code-generating LLMs in production LangChain applications, several factors require attention:

- **Model selection tradeoffs**: While closed-source models such as GPT-4 and Claude demonstrate superior performance on code benchmarks, open-source alternatives such as Llama 3 (70.3% on HumanEval) offer advantages in cost, latency, and data privacy. The appropriate choice depends on specific requirements regarding accuracy, deployment constraints, and budget considerations.
- **Context window management**: Effective handling of limited context windows remains crucial. Recent techniques such as recursive chunking and hierarchical summarization (Li et al., 2024) can improve performance by up to 25% on large codebase tasks.
- **Framework integration** extends basic LLM capabilities by leveraging specialized tools such as LangChain for workflow management. Organizations implementing this pattern establish custom security policies tailored to their domain requirements and build feedback loops that enable continuous improvement of model outputs. This integration approach allows teams to benefit from advances in foundation models while maintaining control over deployment specifics.
- **Human-AI collaboration** establishes clear divisions of responsibility between developers and AI systems. This pattern maintains human oversight for all critical decisions while delegating routine tasks to AI assistants. An essential component is systematic documentation and knowledge capture, ensuring that AI-generated solutions remain comprehensible and maintainable by the entire development team. Companies successfully implementing this pattern report both productivity gains and improved knowledge transfer among team members.

### Security and risk mitigation

When building LLM-powered applications with LangChain, implementing robust security measures and risk mitigation strategies becomes essential. This section focuses on practical approaches to addressing security vulnerabilities, preventing hallucinations, and ensuring code quality through LangChain-specific implementations.

Security vulnerabilities in LLM-generated code present significant risks, particularly when dealing with user inputs, database interactions, or API integrations. LangChain allows developers to create systematic validation processes to identify and mitigate these risks. The following validation chain can be integrated into any LangChain workflow that involves code generation, providing structured security analysis before deployment:

```python
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
# Define the Pydantic model for structured output
class SecurityAnalysis(BaseModel):
    """Security analysis results for generated code."""
    vulnerabilities: List[str] = Field(description="List of identified security vulnerabilities")
   mitigation_suggestions: List[str] = Field(description="Suggested fixes for each vulnerability")
    risk_level: str = Field(description="Overall risk assessment: Low, Medium, High, Critical")

# Initialize the output parser with the Pydantic model
parser = PydanticOutputParser(pydantic_object=SecurityAnalysis)

# Create the prompt template with format instructions from the parser
security_prompt = PromptTemplate.from_template(
    template="""Analyze the following code for security vulnerabilities: {code}
Consider:

SQL injection vulnerabilities
Cross-site scripting (XSS) risks
Insecure direct object references
Authentication and authorization weaknesses
Sensitive data exposure
Missing input validation
Command injection opportunities
Insecure dependency usage
{format_instructions}""",
  input_variables=["code"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Initialize the language model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Compose the chain using LCEL
security_chain = security_prompt | llm | parser
```

The Pydantic output parser ensures that results are properly structured and can be programmatically processed for automated gatekeeping. LLM-generated code should never be directly executed in production environments without validation. LangChain provides tools to create safe execution environments for testing generated code.

To ensure security when building LangChain applications that handle code, a layered approach is crucial, combining LLM-based validation with traditional security tools for robust defense. Structure security findings using Pydantic models and LangChain's output parsers for consistent, actionable outputs. Always isolate the execution of LLM-generated code in sandboxed environments with strict resource limits, never running it directly in production. Explicitly manage dependencies by verifying imports against available packages to avoid hallucinations. Continuously improve code generation through feedback loops incorporating execution results and validation findings. Maintain comprehensive logging of all code generation steps, security findings, and modifications for auditing. Adhere to the principle of least privilege by generating code that follows security best practices such as minimal permissions and proper input validation. Finally, utilize version control to store generated code and implement human review for critical components.

### Validation framework for LLM-generated code

Organizations should implement a structured validation process for LLM-generated code and analyses before moving to production. The following framework provides practical guidance for teams adopting LLMs in their data science workflows:

- **Functional validation** forms the foundation of any assessment process. Start by executing the generated code with representative test data and carefully verify that outputs align with expected results. Ensure all dependencies are properly imported and compatible with your production environment—LLMs occasionally reference outdated or incompatible libraries. Most importantly, confirm that the code actually addresses the original business requirements, as LLMs sometimes produce impressive-looking code that misses the core business objective.
- **Performance assessment** requires looking beyond mere functionality. Benchmark the execution time of LLM-generated code against existing solutions to identify potential inefficiencies. Testing with progressively larger datasets often reveals scaling limitations that weren't apparent with sample data. Profile memory usage systematically, as LLMs may not optimize for resource constraints unless explicitly instructed. This performance data provides crucial information for deployment decisions and identifies opportunities for optimization.
- **Security screening** should never be an afterthought when working with generated code. Scan for unsafe functions, potential injection vulnerabilities, and insecure API calls—issues that LLMs may introduce despite their training in secure coding practices. Verify the proper handling of authentication credentials and sensitive data, especially when the model has been instructed to include API access. Check carefully for hardcoded secrets or unintentional data exposures that could create security vulnerabilities in production.
- **Robustness testing** extends validation beyond the happy path scenarios. Test with edge cases and unexpected inputs that reveal how the code handles extreme conditions. Verify that error handling mechanisms are comprehensive and provide meaningful feedback rather than cryptic failures. Evaluate the code's resilience to malformed or missing data, as production environments rarely provide the pristine data conditions assumed in development.
- **Business logic verification** focuses on domain-specific requirements that LLMs may not fully understand. Confirm that industry-specific constraints and business rules are correctly implemented, especially regulatory requirements that vary by sector. Verify calculations and transformations against manual calculations for critical processes, as subtle mathematical differences can significantly impact business outcomes. Ensure all regulatory or policy requirements relevant to your industry are properly addressed—a crucial step when LLMs may lack domain-specific compliance knowledge.
- **Documentation and explainability** complete the validation process by ensuring sustainable use of the generated code. Either require the LLM to provide or separately generate inline comments that explain complex sections and algorithmic choices. Document any assumptions made by the model that might impact future maintenance or enhancement. Create validation reports that link code functionality directly to business requirements, providing traceability that supports both technical and business stakeholders.

This validation framework should be integrated into development workflows, with appropriate automation incorporated where possible to reduce manual effort. Organizations embarking on LLM adoption should start with well-defined use cases clearly aligned with business objectives, implement these validation processes systematically, invest in comprehensive staff training on both LLM capabilities and limitations, and establish clear governance frameworks that evolve with the technology.

### LangChain integrations

As we're aware, LangChain enables the creation of versatile and robust AI agents. For instance, a LangChain-integrated agent can safely execute code using dedicated interpreters, interact with SQL databases for dynamic data retrieval, and perform real-time financial analysis, all while upholding strict quality and security standards.

Integrations range from code execution and database querying to financial analysis and repository management. This wide-ranging toolkit facilitates building applications that are deeply integrated with real-world data and systems, ensuring that AI solutions are both powerful and practical. Here are some examples of integrations:

- **Code execution and isolation: **Tools such as the Python REPL, Azure Container Apps dynamic sessions, Riza Code Interpreter, and Bearly Code Interpreter provide various environments to safely execute code. They enable LLMs to delegate complex calculations or data processing tasks to dedicated code interpreters, thereby increasing accuracy and reliability while maintaining security.
- **Database and data handling: **Integrations for Cassandra, SQL, and Spark SQL toolkits allow agents to interface directly with different types of databases. Meanwhile, JSON Toolkit and pandas DataFrame integration facilitate efficient handling of structured data. These capabilities are essential for applications that require dynamic data retrieval, transformation, and analysis.
- **Financial data and analysis: **With FMP Data, Google Finance, and the FinancialDatasets Toolkit, developers can build AI agents capable of performing sophisticated financial analyses and market research. Dappier further extends this by connecting agents to curated, real-time data streams.
- **Repository and version control integration: **The GitHub and GitLab toolkits enable agents to interact with code repositories, streamlining tasks such as issue management, code reviews, and deployment processes—a crucial asset for developers working in modern DevOps environments.
- **User input and visualization: **Google Trends and PowerBI Toolkit highlight the ecosystem's focus on bringing in external data (such as market trends) and then visualizing it effectively. The "human as a tool" integration is a reminder that, sometimes, human judgment remains indispensable, especially in ambiguous scenarios.

Having explored the theoretical framework and potential benefits of LLM-assisted software development, let's now turn to practical implementation. In the following section, we'll demonstrate how to generate functional software code with LLMs and execute it directly from within the LangChain framework. This hands-on approach will illustrate the concepts we've discussed and provide you with actionable examples you can adapt to your own projects.

## Writing code with LLMs

In this section, we demonstrate code generation using various models integrated with LangChain. We've selected different models to showcase:

- LangChain's diverse integrations with AI tools
- Models with different licensing and availability
- Options for local deployment, including smaller models

These examples illustrate LangChain's flexibility in working with various code generation models, from cloud-based services to open-source alternatives. This approach allows you to understand the range of options available and choose the most suitable solution for your specific needs and constraints.

> Please make sure you have installed all the dependencies needed for this book, as explained in [_Chapter 2_](Chapter_2.xhtml#_idTextAnchor025). Otherwise, you might run into issues.
>
> Given the pace of the field and the development of the LangChain library, we are making an effort to keep the GitHub repository up to date. Please see [https://github.com/benman1/generative_ai_with_langchain](https://github.com/benman1/generative_ai_with_langchain).
>
> For any questions or if you have any trouble running the code, please create an issue on GitHub or join the discussion on Discord: [https://packt.link/lang](https://packt.link/lang).

### Google generative AI

The Google generative AI platform offers a range of models designed for instruction following, conversion, and code generation/assistance. These models also have different input/output limits and training data and are often updated. Let's see if the Gemini Pro model can solve **FizzBuzz**, a common interview question for entry-level software developer positions.

To test the model's code generation capabilities, we'll use LangChain to interface with Gemini Pro and provide the FizzBuzz problem statement:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
question = """

Given an integer n, return a string array answer (1-indexed) where:
answer[i] == "FizzBuzz" if i is divisible by 3 and 5.
answer[i] == "Fizz" if i is divisible by 3.
answer[i] == "Buzz" if i is divisible by 5.
answer[i] == i (as a string) if none of the above conditions are true.
"""

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
print(llm.invoke(question).content)
```

Gemini Pro immediately returns a clean, correct Python solution that properly handles all the FizzBuzz requirements:

```python
    answer = []

    for i in range(1, n+1):
        if i % 3 == 0 and i % 5 == 0:
            answer.append("FizzBuzz")
        elif i % 3 == 0:
            answer.append("Fizz")
        elif i % 5 == 0:
            answer.append("Buzz")
        else:
            answer.append(str(i))

    return answer
```

The model produced an efficient, well-structured solution that correctly implements the logic for the FizzBuzz problem without any errors or unnecessary complexity. Would you hire Gemini Pro for your team?

### Hugging Face

Hugging Face hosts a lot of open-source models, many of which have been trained on code, some of which can be tried out in playgrounds, where you can ask them to either complete (for older models) or write code (instruction-tuned models). With LangChain, you can either download these models and run them locally, or you can access them through the Hugging Face API. Let's try the local option first with a prime number calculation example:

```python
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Choose a more up-to-date model
checkpoint = "google/codegemma-2b"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Create a text generation pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500
)

# Integrate the pipeline with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Define the input text
text = """

def calculate_primes(n):
    """Create a list of consecutive integers from 2 up to N.
    For example:
    >>> calculate_primes(20)
    Output: [2, 3, 5, 7, 11, 13, 17, 19]
    """
"""

# Use the LangChain LLM to generate text
output = llm(text)
print(output)
```

When executed, CodeGemma completes the function by implementing the Sieve of Eratosthenes algorithm, a classic method for finding prime numbers efficiently. The model correctly interprets the docstring, understanding that the function should return all prime numbers up to n rather than just checking whether a number is prime. The generated code demonstrates how specialized code models can produce working implementations from minimal specifications.

> Please note that the downloading and loading of the models can take a few minutes.

If you're getting an error saying you "`cannot access a gated repo`" when trying to use a URL with LangChain, it means you're attempting to access a private repository on Hugging Face that requires authentication with a personal access token to view or use the model; you need to create a Hugging Face access token and set it as an environment variable named `"HF_TOKEN"` to access the gated repository. You can get the token on the Hugging Face website at [https://huggingface.co/docs/api-inference/quicktour#get-your-api-token](https://huggingface.co/docs/api-inference/quicktour#get-your-api-token).

When our code from the previous example executes successfully with CodeGemma, it generates a complete implementation for the prime number calculator function. The output looks like this:

```python
def calculate_primes(n):
    """Create a list of consecutive integers from 2 up to N.
    For example:
    >>> calculate_primes(20)
    Output: [2, 3, 5, 7, 11, 13, 17, 19]
    """
    primes = []
    for i in range(2, n + 1):
        if is_prime(i):
            primes.append(i)
    return primes

def is_prime(n):
    """Return True if n is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def main():
    """Get user input and print the list of primes."""
    n = int(input("Enter a number: "))
    primes = calculate_primes(n)
    print(primes)

if __name__ == "__main__":
    main()
<|file_separator|>
```

Notice how the model not only implemented the requested `calculate_primes()` function but also created a helper function, `is_prime()`, which uses a more efficient algorithm checking divisibility only up to the square root of the number. The model even added a complete `main()` function with user input handling, demonstrating its understanding of Python programming patterns.

Instead of downloading and running models locally, which requires significant computational resources, we can also run models directly on Hugging Face's infrastructure using their Inference API. This approach is simpler to set up and doesn't require powerful hardware. Here's how to implement the same example using Hugging Face's hosted services:

```python
from langchain.llms import HuggingFaceHub

# Choose a lightweight model good for code generation
repo_id = "bigcode/starcoder"

# Initialize the HuggingFaceHub LLM
llm = HuggingFaceHub(
    repo_id=repo_id,
    task="text-generation",
    model_kwargs={
        "temperature": 0.5,
        "max_length": 1000
    }
)

# Use the LangChain LLM to generate text
output = llm.invoke(text)
print(output)
```

When executed, this code connects to Hugging Face's servers to run the StarCoder model, a specialized code generation model trained on a vast corpus of source code. The expected output would be similar to our previous example—a complete implementation of the prime number calculator—but potentially with different algorithmic approaches since we're using a different model. This hosted approach trades some flexibility and control for convenience and reduced local resource requirements, making it ideal for quick prototyping or when working on hardware with limited capabilities.

### Anthropic

Anthropic Claude series models have been particularly good in coding-related tasks. Let's see how Claude does at a coding task:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts.prompt import PromptTemplate

template = """Question: {question}

Let's think step by step.
Answer:

"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = ChatAnthropic(model='claude-3-opus-20240229')
llm_chain = prompt | llm
print(llm_chain.invoke(text))
```

We get an algorithm and a very detailed explanation:

````python
To calculate the prime numbers up to a given number N, we can follow the Sieve of Eratosthenes algorithm. Here's a step-by-step approach to solve the problem:

```python
def calculate_primes(n):
    """Create a list of consecutive integers from 2 up to N."""
    # Create a list of consecutive integers from 2 to N
    numbers = list(range(2, n + 1))
    # Initialize an empty list to store the prime numbers
    primes = []
    while numbers:
        # Take the first number as prime
        prime = numbers[0]
        primes.append(prime)
        # Remove all multiples of the prime number from the list
        numbers = [num for num in numbers if num % prime != 0]
    return primes
````

Step-by-step explanation:
We create a list called `numbers` that contains consecutive integers from 2 to N using the `range()` function.
We initialize an empty list called `primes` to store the prime numbers that we will calculate.
We start a loop that continues as long as there are numbers in the `numbers` list.
Inside the loop, we take the first number from the `numbers` list and consider it as a prime number. We append this prime number to the `primes` list.
We remove all the multiples of the prime number from the `numbers` list using a list comprehension. The list comprehension creates a new list containing only the numbers that are not divisible by the current prime number.
We repeat steps 4 and 5 until the `numbers` list becomes empty. This ensures that we have considered all the numbers up to N and identified the prime numbers among them.
Finally, we return the `primes` list, which contains all the prime numbers up to N.

````

Let's test the function with the given example:

```python
print(calculate_primes(20))
````

```
Output:

[2, 3, 5, 7, 11, 13, 17, 19]
```

The function correctly returns the list of prime numbers up to 20. This implementation follows the Sieve of Eratosthenes algorithm again.

### Agentic approach

We can also create an LLM agent that can execute Python code to solve problems:

```python
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_experimental.tools import PythonREPLTool

tools = [PythonREPLTool()]   # Gives agent ability to run Python code
llm = ChatOpenAI()

# Set up the agent with necessary tools and model
agent = initialize_agent(
    tools,
    llm,  # Language model to power the agent
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # Shows agent's thinking process
)  # Agent makes decisions without examples

result = agent("What are the prime numbers until 20?")
print(result)
```

The agent will:

1. Determine what it needs to write Python code.
2. Use `PythonREPLTool` to execute the code.
3. Return the results.

When run, it will show its reasoning steps and code execution before giving the final answer. We should be seeing an output like this:

```
> Entering new AgentExecutor chain...
I can write a Python script to find the prime numbers up to 20.
Action: Python_REPL
Action Input: def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
primes = [num for num in range(2, 21) if is_prime(num)]
print(primes)
Observation: [2, 3, 5, 7, 11, 13, 17, 19]
I now know the final answer
Final Answer: [2, 3, 5, 7, 11, 13, 17, 19]
> Finished chain.
{'input': 'What are the prime numbers until 20?', 'output': '[2, 3, 5, 7, 11, 13, 17, 19]'}
```

### Documentation RAG

What is also quite interesting is the use of documents to help write code or to ask questions about documentation. Here's an example of loading all documentation pages from LangChain's website using `DocusaurusLoader`:

```python
from langchain_community.document_loaders import DocusaurusLoader
import nest_asyncio
nest_asyncio.apply()

# Load all pages from LangChain docs
loader = DocusaurusLoader("https://python.langchain.com")
documents[0]

nest_asyncio.apply() enables async operations in Jupyter notebooks. The loader gets all pages.
```

`DocusaurusLoader` automatically scrapes and extracts content from LangChain's documentation website. This loader is specifically designed to navigate Docusaurus-based sites and extract properly formatted content. Meanwhile, the `nest_asyncio.apply()` function is necessary for a Jupyter Notebook environment, which has limitations with asyncio's event loop. This line allows us to run asynchronous code within the notebook's cells, which is required for many web-scraping operations. After execution, the documents variable contains all the documentation pages, each represented as a `Document` object with properties like `page_content` and metadata. We can then set up embeddings with caching:

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore

# Cache embeddings locally to avoid redundant API calls
store = LocalFileStore("./cache/")
underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)
```

Before we can feed our models into a vector store, we need to split them, as discussed in [_Chapter 4_](Chapter_4.xhtml#_idTextAnchor068):

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

splits = text_splitter.split_documents(documents)
```

Now we'll create a vector store from the document splits:

```python
from langchain_chroma import Chroma

# Store document embeddings for efficient retrieval
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
```

We'll also need to initialize the LLM or chat model:

```python
from langchain_google_vertexai import VertexAI
llm = VertexAI(model_name="gemini-pro")
```

Then, we set up the RAG components:

```python
from langchain import hub
retriever = vectorstore.as_retriever()

# Use community-created RAG prompt template
prompt = hub.pull("rlm/rag-prompt")
```

Finally, we'll build the RAG chain:

```python
from langchain_core.runnables import RunnablePassthrough
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain combines context retrieval, prompting, and response generation
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Let's query the chain:

```python
response = rag_chain.invoke("What is Task Decomposition?")
```

Each component builds on the previous one, creating a complete RAG system that can answer questions using the LangChain documentation.

### Repository RAG

One powerful application of RAG systems is analyzing code repositories to enable natural language queries about codebases. This technique allows developers to quickly understand unfamiliar code or find relevant implementation examples. Let's build a code-focused RAG system by indexing a GitHub repository.

First, we'll clone the repository and set up our environment:

```python
import os
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Clone the book repository from GitHub
repo_path = os.path.expanduser("~/Downloads/generative_ai_with_langchain")  # this directory should not exist yet!
repo = Repo.clone_from("https://github.com/benman1/generative_ai_with_langchain", to_path=repo_path)
```

After cloning the repository, we need to parse the Python files using LangChain's specialized loaders that understand code structure. LanguageParser helps maintain code semantics during processing:

```python
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)

# Split the Document into chunks for embedding and vector storage
texts = python_splitter.split_documents(documents)
```

This code performs three key operations: it clones our book's GitHub repository, loads all Python files using language-aware parsing, and splits the code into smaller, semantically meaningful chunks. The language-specific splitter ensures we preserve function and class definitions when possible, making our retrieval more effective.

Now we'll create our RAG system by embedding these code chunks and setting up a retrieval chain:

```python
# Create vector store and retriever
db = Chroma.from_documents(texts, OpenAIEmbeddings())
retriever = db.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance for diverse results
    search_kwargs={"k": 8}  # Return 8 most relevant chunks
)

# Set up Q&A chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context:\n\n{context}"),
    ("placeholder", "{chat_history}"),
    ("user", "{input}"),
])

# Create chain components
document_chain = create_stuff_documents_chain(ChatOpenAI(), prompt)
qa = create_retrieval_chain(retriever, document_chain)
```

Here, we've built our complete RAG pipeline: we store code embeddings in a Chroma vector database, configure a retriever to use maximal marginal relevance (which helps provide diverse results), and create a QA chain that combines retrieved code with our prompt template before sending it to the LLM.

Let's test our code-aware RAG system with a question about software development examples:

```python
question = "What examples are in the code related to software development?"
result = qa.invoke({"input": question})
print(result["answer"])
Here are some examples of the code related to software development in the given context:
1. Task planner and executor for software development: This indicates that the code includes functionality for planning and executing tasks related to software development.
2. debug your code: This suggests that there is a recommendation to debug the code if an error occurs during software development.
These examples provide insights into the software development process described in the context.
```

The response is somewhat limited, likely because our small chunk size (50 characters) may have fragmented code examples. While the system correctly identifies mentions of task planning and debugging, it doesn't provide detailed code examples or context. In a production environment, you might want to increase the chunk size or implement hierarchical chunking to preserve more context. Additionally, using a code-specific embedding model could further improve the relevance of retrieved results.

In the next section, we'll explore how generative AI agents can automate and enhance data science workflows. LangChain agents can write and execute code, analyze datasets, and even build and train ML models with minimal human guidance. We'll demonstrate two powerful applications: training a neural network model and analyzing a structured dataset.

## Applying LLM agents for data science

The integration of LLMs into data science workflows represents a significant, though nuanced, evolution in how analytical tasks are approached. While traditional data science methods remain essential for complex numerical analysis, LLMs offer complementary capabilities that primarily enhance accessibility and assist with specific aspects of the workflow.

Independent research reveals a more measured reality than some vendor claims suggest. According to multiple studies, LLMs demonstrate variable effectiveness across different data science tasks, with performance often declining as complexity increases. A study published in PLOS One found that "the executability of generated code decreased significantly as the complexity of the data analysis task increased," highlighting the limitations of current models when handling sophisticated analytical challenges.

LLMs exhibit a fundamental distinction in their data focus compared to traditional methods. While traditional statistical techniques excel at processing structured, tabular data through well-defined mathematical relationships, LLMs demonstrate superior capabilities with unstructured text. They can generate code for common data science tasks, particularly boilerplate operations involving data manipulation, visualization, and routine statistical analyses. Research on GitHub Copilot and similar tools indicates that these assistants can meaningfully accelerate development, though the productivity gains observed in independent studies (typically 7–22%) are more modest than some vendors claim. BlueOptima's analysis of over 218,000 developers found productivity improvements closer to 4% rather than the 55% claimed in controlled experiments.

Text-to-SQL capabilities represent one of the most promising applications, potentially democratizing data access by allowing non-technical users to query databases in natural language. However, the performance often drops on the more realistic BIRD benchmark compared to Spider, and accuracy remains a key concern, with performance varying significantly based on the complexity of the query, the database schema, and the benchmark used.

LLMs also excel at translating technical findings into accessible narratives for non-technical audiences, functioning as a communication bridge in data-driven organizations. While systems such as InsightLens demonstrate automated insight organization capabilities, the technology shows clear strengths and limitations when generating different types of content. The contrast is particularly stark with synthetic data: LLMs effectively create qualitative text samples but struggle with structured numerical datasets requiring complex statistical relationships. This performance boundary aligns with their core text processing capabilities and highlights where traditional statistical methods remain superior. A study published in JAMIA (_Evaluating Large Language Models for Health-Related Text Classification Tasks with Public Social Media Data_, 2024) found that "LLMs (specifically GPT-4, but not GPT-3.5) [were] effective for data augmentation in social media health text classification tasks but ineffective when used alone to annotate training data for supervised models."

The evidence points toward a future where LLMs and traditional data analysis tools coexist and complement each other. The most effective implementations will likely be hybrid systems leveraging:

- LLMs for natural language interaction, code assistance, text processing, and initial exploration
- Traditional statistical and ML techniques for rigorous analysis of structured data and high-stakes prediction tasks

The transformation brought by LLMs enables both technical and non-technical stakeholders to interact with data effectively. Its primary value lies in reducing the cognitive load associated with repetitive coding tasks, allowing data scientists to maintain the flow and focus on higher-level analytical challenges. However, rigorous validation remains essential—independent studies consistently identify concerns regarding code quality, security, and maintainability. These considerations are especially critical in two key workflows that LangChain has revolutionized: training ML models and analyzing datasets.

When training ML models, LLMs can now generate synthetic training data, assist in feature engineering, and automatically tune hyperparameters—dramatically reducing the expertise barrier for model development. Moreover, for data analysis, LLMs serve as intelligent interfaces that translate natural language questions into code, visualizations, and insights, allowing domain experts to extract value from data without deep programming knowledge. The following sections explore both of these areas with LangChain.

### Training an ML model

As you know by now, LangChain agents can write and execute Python code for data science tasks, including building and training ML models. This capability is particularly valuable when you need to perform complex data analysis, create visualizations, or implement custom algorithms on the fly without switching contexts.

In this section, we'll explore how to create and use Python-capable agents through two main steps: setting up the Python agent environment and configuring the agent with the right model and tools; and implementing a neural network from scratch, guiding the agent to create a complete working model.

#### Setting up a Python-capable agent

Let's start by creating a Python-capable agent using LangChain's experimental tools:

```python
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_anthropic import ChatAnthropic
from langchain.agents.agent_types import AgentType

agent_executor = create_python_agent(
    llm=ChatAnthropic(model='claude-3-opus-20240229'),
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
```

This code creates a Python agent with the Claude 3 Opus model, which offers strong reasoning capabilities for complex programming tasks. `PythonREPLTool` provides the agent with a Python execution environment, allowing it to write and run code, see outputs, and iterate based on results. Setting `verbose=True` lets us observe the agent's thought process, which is valuable for understanding its approach and debugging.

> **Security caution**
>
> PythonREPLTool executes arbitrary Python code with the same permissions as your application. While excellent for development and demonstrations, this presents significant security risks in production environments. For production deployments, consider:
>
> - Using restricted execution environments such as RestrictedPython or Docker containers
> - Implementing custom tools with explicit permission boundaries
> - Running the agent in a separate isolated service with limited permissions
> - Adding validation and sanitization steps before executing generated code

The `AgentExecutor`, on the other hand, is a LangChain component that orchestrates the execution loop for agents. It manages the agent's decision-making process, handles interactions with tools, enforces iteration limits, and processes the agent's final output. Think of it as the runtime environment where the agent operates.

#### Asking the agent to build a neural network

Now that we've set up our Python agent, let's test its capabilities with a practical ML task. We'll challenge the agent to implement a simple neural network that learns a basic linear relationship. This example demonstrates how agents can handle end-to-end ML development tasks from data generation to model training and evaluation.

The following code instructs our agent to create a single-neuron neural network in PyTorch, train it on synthetic data representing the function `y=2x`, and make a prediction:

```python
result = agent_executor.run(
    """Understand, write a single neuron neural network in PyTorch.
Take synthetic data for y=2x. Train for 1000 epochs and print every 100 epochs.
Return prediction for x = 5"""
)
print(result)
```

This concise prompt instructs the agent to implement a full neural network pipeline: generating PyTorch code for a single-neuron model, creating synthetic training data that follows `y=2x`, training the model over 1,000 epochs with periodic progress reports, and, finally, making a prediction for a new input value of `x=5`.

#### Agent execution and results

When we run this code, the agent begins reasoning through the problem and executing Python code. Here's the abbreviated verbose output showing the agent's thought process and execution:

```
> Entering new AgentExecutor chain...
Here is a single neuron neural network in PyTorch that trains on synthetic data for y=2x, prints the loss every 100 epochs, and returns the prediction for x=5:
Action: Python_REPL
Action Input:
import torch
import torch.nn as nn
# Create synthetic data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Define the model
[...] # Code for creating the model omitted for brevity
Observation:
Epoch [100/1000], Loss: 0.0529
[...] # Training progress for epochs 200-900 omitted for brevity
Epoch [1000/1000], Loss: 0.0004
Prediction for x=5: 9.9659
To summarize:
- I created a single neuron neural network model in PyTorch using nn.Linear(1, 1)
- I generated synthetic data where y=2x for training
- I defined the MSE loss function and SGD optimizer
- I trained the model for 1000 epochs, printing the loss every 100 epochs
- After training, I made a prediction for x=5
The final prediction for x=5 is 9.9659, which is very close to the expected value of 10 (since y=2x).
So in conclusion, I was able to train a simple single neuron PyTorch model to fit the synthetic y=2x data well and make an accurate prediction for a new input x=5.
Final Answer: The trained single neuron PyTorch model predicts a value of 9.9659 for x=5.
> Finished chain.
The final output confirms that our agent successfully built and trained a model that learned the y=2x relationship. The prediction for x=5 is approximately 9.97, which is very close to the expected value of 10.
```

The results demonstrate that our agent successfully built and trained a neural network. The prediction for x=5 is approximately 9.97, very close to the expected value of 10 (since 2×5=10). This accuracy confirms that the model effectively learned the underlying linear relationship from our synthetic data.

> If your agent produces unsatisfactory results, consider increasing specificity in your prompt (e.g., specify learning rate or model architecture), requesting validation steps such as plotting the loss curve, lowering the LLM temperature for more deterministic results, or breaking complex tasks into sequential prompts.

This example showcases how LangChain agents can successfully implement ML workflows with minimal human intervention. The agent demonstrated strong capabilities in understanding the requested task, generating correct PyTorch code without reference examples, creating appropriate synthetic data, configuring and training the neural network, and evaluating results against expected outcomes.

In a real-world scenario, you could extend this approach to more complex ML tasks such as classification problems, time series forecasting, or even custom model architectures. Next, we'll explore how agents can assist with data analysis and visualization tasks that build upon these fundamental ML capabilities.

### Analyzing a dataset

Next, we'll demonstrate how LangChain agents can analyze structured datasets by examining the well-known `Iris` dataset. The `Iris` dataset, created by British statistician Ronald Fisher, contains measurements of sepal length, sepal width, petal length, and petal width for three species of iris flowers. It's commonly used in machine learning for classification tasks.

#### Creating a pandas DataFrame agent

Data analysis is a perfect application for LLM agents. Let's explore how to create an agent specialized in working with pandas DataFrames, enabling natural language interaction with tabular data.

First, we'll load the classic Iris dataset and save it as a CSV file for our agent to work with:

```python
from sklearn.datasets import load_iris
df = load_iris(as_frame=True)["data"]
df.to_csv("iris.csv", index=False)
```

Now we'll create a specialized agent for working with pandas DataFrames:

```python
from langchain_experimental.agents.agent_toolkits.pandas.base import
create_pandas_dataframe_agent
from langchain import PromptTemplate

PROMPT = (
    "If you do not know the answer, say you don't know.\n"
    "Think step by step.\n"
    "\n"
    "Below is the query.\n"
    "Query: {query}\n"
)

prompt = PromptTemplate(template=PROMPT, input_variables=["query"])
llm = OpenAI()

agent = create_pandas_dataframe_agent(
    llm, df, verbose=True, allow_dangerous_code=True
)
```

> **Security warning**
>
> We've used `allow_dangerous_code=True`, which permits the agent to execute any Python code on your machine. This could potentially be harmful if the agent generates malicious code. Only use this option in development environments with trusted data sources, and never in production scenarios without proper sandboxing.

The example above works well with small datasets like Iris (150 rows), but real-world data analysis often involves much larger datasets that exceed LLM context windows. When implementing DataFrame agents in production environments, several strategies can help overcome these limitations.

Data summarization and preprocessing techniques form your first line of defense. Before sending data to your agent, consider extracting key statistical information such as shape, column names, data types, and summary statistics (mean, median, max, etc.). Including representative samples—perhaps the first and last few rows or a small random sample—provides context without overwhelming the LLM's token limit. This preprocessing approach preserves critical information while dramatically reducing the input size.

For datasets that are too large for a single context window, chunking strategies offer an effective solution. You can process the data in manageable segments, run your agent on each chunk separately, and then aggregate the results. The aggregation logic would depend on the specific analysis task—for example, finding global maximums across chunk-level results for optimization queries or combining partial analyses for more complex tasks. This approach trades some global context for the ability to handle datasets of any size.

Query-specific preprocessing adapts your approach based on the nature of the question. Statistical queries can often be pre-aggregated before sending to the agent. For correlation questions, calculating and providing the correlation matrix upfront helps the LLM focus on interpretation rather than computation. For exploratory questions, providing dataset metadata and samples may be sufficient. This targeted preprocessing makes efficient use of context windows by including only relevant information for each specific query type.

#### Asking questions about the dataset

Now that we've set up our data analysis agent, let's explore its capabilities by asking progressively complex questions about our dataset. A well-designed agent should be able to handle different types of analytical tasks, from basic exploration to statistical analysis and visualization. The following examples demonstrate how our agent can work with the classic Iris dataset, which contains measurements of flower characteristics.

We'll test our agent with three types of queries that represent common data analysis workflows: understanding the data structure, performing statistical calculations, and creating visualizations. These examples showcase the agent's ability to reason through problems, execute appropriate code, and provide useful answers.

First, let's ask a fundamental exploratory question to understand what data we're working with:

```python
agent.run(prompt.format(query="What's this dataset about?"))
```

The agent executes this request by examining the dataset structure:

```
Output:
> Entering new AgentExecutor chain...
Thought: I need to understand the structure and contents of the dataset.
Action: python_repl_ast
Action Input: print(df.head())
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2
 This dataset contains four features (sepal length, sepal width, petal length, and petal width) and 150 entries.
Final Answer: Based on the observation, this dataset is likely about measurements of flower characteristics.
> Finished chain.
'Based on the observation, this dataset is likely about measurements of flower characteristics.'
```

This initial query demonstrates how the agent can perform basic data exploration by checking the structure and first few rows of the dataset. Notice how it correctly identifies that the data contains flower measurements, even without explicit species labels in the preview. Next, let's challenge our agent with a more analytical question that requires computation:

```python
agent.run(prompt.format(query="Which row has the biggest difference between petal length and petal width?"))
```

The agent tackles this by creating a new calculated column and finding its maximum value:

```
> Entering new AgentExecutor chain...
Thought: First, we need to find the difference between petal length and petal width for each row. Then, we need to find the row with the maximum difference.
Action: python_repl_ast
Action Input: df['petal_diff'] = df['petal length (cm)'] - df['petal width (cm)']
              df['petal_diff'].max()
Observation: 4.7
Action: python_repl_ast
Action Input: df['petal_diff'].idxmax()
Observation: 122
Final Answer: Row 122 has the biggest difference between petal length and petal width.
> Finished chain.
'Row 122 has the biggest difference between petal length and petal width.'
```

This example shows how our agent can perform more complex analysis by:

- Creating derived metrics (the difference between two columns)
- Finding the maximum value of this metric
- Identifying which row contains this value

Finally, let's see how our agent handles a request for data visualization:

```python
agent.run(prompt.format(query="Show the distributions for each column visually!"))
```

For this visualization query, the agent generates code to create appropriate plots for each measurement column. The agent decides to use histograms to show the distribution of each feature in the dataset, providing visual insights that complement the numerical analyses from previous queries. This demonstrates how our agent can generate code for creating informative data visualizations that help understand the dataset's characteristics.

These three examples showcase the versatility of our data analysis agent in handling different types of analytical tasks. By progressively increasing the complexity of our queries—from basic exploration to statistical analysis and visualization—we can see how the agent uses its tools effectively to provide meaningful insights about the data.

> When designing your own data analysis agents, consider providing them with a variety of analysis tools that cover the full spectrum of data science workflows: exploration, preprocessing, analysis, visualization, and interpretation.

![Figure 7.2: Our LLM agent visualizing the well-known Iris dataset](Images/B32363_07_02.png)

In the repository, you can see a UI that wraps a data science agent.

Data science agents represent a powerful application of LangChain's capabilities. These agents can:

- Generate and execute Python code for data analysis and machine learning
- Build and train models based on simple natural language instructions
- Answer complex questions about datasets through analysis and visualization
- Automate repetitive data science tasks

While these agents aren't yet ready to replace human data scientists, they can significantly accelerate workflows by handling routine tasks and providing quick insights from data.

Let's conclude the chapter!

## Summary

This chapter has examined how LLMs are reshaping software development and data analysis practices through natural language interfaces. We traced the evolution from early code generation models to today's sophisticated systems, analyzing benchmarks that reveal both capabilities and limitations. Independent research suggests that while 55% productivity gains in controlled settings don't fully translate to production environments, meaningful improvements of 4-22% are still being realized, particularly when human expertise guides LLM implementation.

Our practical demonstrations illustrated diverse approaches to LLM integration through LangChain. We used multiple models to generate code solutions, built RAG systems to augment LLMs with documentation and repository knowledge, and created agents capable of training neural networks and analyzing datasets with minimal human intervention. Throughout these implementations, we looked at critical security considerations, providing validation frameworks and risk mitigation strategies essential for production deployments.

Having explored the capabilities and integration strategies for LLMs in software and data workflows, we now turn our attention to ensuring these solutions work reliably in production. In [_Chapter 8_](Chapter_8.xhtml#_idTextAnchor157), we'll delve into evaluation and testing methodologies that help validate AI-generated code and safeguard system performance, setting the stage for building truly production-ready applications.

## Questions

1. What is vibe coding, and how does it change the traditional approach to writing and maintaining code?
2. What key differences exist between traditional low-code platforms and LLM-based development approaches?
3. How do independent research findings on productivity gains from AI coding assistants differ from vendor claims, and what factors might explain this discrepancy?
4. What specific benchmark metrics show that LLMs struggle more with class-level code generation compared to function-level tasks, and why is this distinction important for practical implementations?
5. Describe the validation framework presented in the chapter for LLM-generated code. What are the six key areas of assessment, and why is each important for production systems?
6. Using the repository RAG example from the chapter, explain how you would modify the implementation to better handle large codebases with thousands of files.
7. What patterns emerged in the dataset analysis examples that demonstrate how LLMs perform in structured data analysis tasks versus unstructured text processing?
8. How does the agentic approach to data science, as demonstrated in the neural network training example, differ from traditional programming workflows? What advantages and limitations did this approach reveal?
9. How do LLM integrations in LangChain enable more effective software development and data analysis?
10. What critical factors should organizations consider when implementing LLM-based development or analysis tools?
