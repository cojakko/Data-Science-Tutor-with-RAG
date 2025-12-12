# LangChain overview

<Callout icon="bullhorn" color="#DFC5FE" iconType="regular">
  **LangChain v1.0 is now available!**

  For a complete list of changes and instructions on how to upgrade your code, see the [release notes](/oss/python/releases/langchain-v1) and [migration guide](/oss/python/migrate/langchain-v1).

  If you encounter any issues or have feedback, please [open an issue](https://github.com/langchain-ai/docs/issues/new?template=01-langchain.yml) so we can improve. To view v0.x documentation, [go to the archived content](https://github.com/langchain-ai/langchain/tree/v0.3/docs/docs).
</Callout>

LangChain is the easiest way to start building agents and applications powered by LLMs. With under 10 lines of code, you can connect to OpenAI, Anthropic, Google, and [more](/oss/python/integrations/providers/overview). LangChain provides a pre-built agent architecture and model integrations to help you get started quickly and seamlessly incorporate LLMs into your agents and applications.

We recommend you use LangChain if you want to quickly build agents and autonomous applications. Use [LangGraph](/oss/python/langgraph/overview), our low-level agent orchestration framework and runtime, when you have more advanced needs that require a combination of deterministic and agentic workflows, heavy customization, and carefully controlled latency.

LangChain [agents](/oss/python/langchain/agents) are built on top of LangGraph in order to provide durable execution, streaming, human-in-the-loop, persistence, and more. You do not need to know LangGraph for basic LangChain agent usage.

## <Icon icon="download" size={20} /> Install

<CodeGroup>
  ```bash pip theme={null}
  pip install -U langchain
  ```

  ```bash uv theme={null}
  uv add langchain
  ```
</CodeGroup>

## <Icon icon="wand-magic-sparkles" /> Create an agent

```python  theme={null}
# pip install -qU "langchain[anthropic]" to call the model

from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

## <Icon icon="star" size={20} /> Core benefits

<Columns cols={2}>
  <Card title="Standard model interface" icon="arrows-rotate" href="/oss/python/langchain/models" arrow cta="Learn more">
    Different providers have unique APIs for interacting with models, including the format of responses. LangChain standardizes how you interact with models so that you can seamlessly swap providers and avoid lock-in.
  </Card>

  <Card title="Easy to use, highly flexible agent" icon="wand-magic-sparkles" href="/oss/python/langchain/agents" arrow cta="Learn more">
    LangChain's agent abstraction is designed to be easy to get started with, letting you build a simple agent in under 10 lines of code. But it also provides enough flexibility to allow you to do all the context engineering your heart desires.
  </Card>

  <Card title="Built on top of LangGraph" icon="circle-nodes" href="/oss/python/langgraph/overview" arrow cta="Learn more">
    LangChain's agents are built on top of LangGraph. This allows us to take advantage of LangGraph's durable execution, human-in-the-loop support, persistence, and more.
  </Card>

  <Card title="Debug with LangSmith" icon="eye" href="/langsmith/home" arrow cta="Learn more">
    Gain deep visibility into complex agent behavior with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics.
  </Card>
</Columns>

***

<Callout icon="pen-to-square" iconType="regular">
  [Edit the source of this page on GitHub.](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/overview.mdx)
</Callout>

<Tip icon="terminal" iconType="regular">
  [Connect these docs programmatically](/use-these-docs) to Claude, VSCode, and more via MCP for    real-time answers.
</Tip>
LangChain Expression Language (LCEL)
Prerequisites
Runnable Interface
The LangChain Expression Language (LCEL) takes a declarative approach to building new Runnables from existing Runnables.

This means that you describe what you want to happen, rather than how you want it to happen, allowing LangChain to optimize the run-time execution of the chains.

We often refer to a Runnable created using LCEL as a "chain". It's important to remember that a "chain" is Runnable and it implements the full Runnable Interface.

note
The LCEL cheatsheet shows common patterns that involve the Runnable interface and LCEL expressions.
Please see the following list of how-to guides that cover common tasks with LCEL.
A list of built-in Runnables can be found in the LangChain Core API Reference. Many of these Runnables are useful when composing custom "chains" in LangChain using LCEL.
Benefits of LCEL
LangChain optimizes the run-time execution of chains built with LCEL in a number of ways:

Optimize parallel execution: Run Runnables in parallel using RunnableParallel or run multiple inputs through a given chain in parallel using the Runnable Batch API. Parallel execution can significantly reduce the latency as processing can be done in parallel instead of sequentially.
Simplify streaming: LCEL chains can be streamed, allowing for incremental output as the chain is executed. LangChain can optimize the streaming of the output to minimize the time-to-first-token(time elapsed until the first chunk of output from a chat model or llm comes out).
Other benefits include:

Seamless LangSmith tracing As your chains get more and more complex, it becomes increasingly important to understand what exactly is happening at every step. With LCEL, all steps are automatically logged to LangSmith for maximum observability and debuggability.
Standard API: Because all chains are built using the Runnable interface, they can be used in the same way as any other Runnable.
Deployable with LangServe: Chains built with LCEL can be deployed using for production use.
Should I use LCEL?
LCEL is an orchestration solution -- it allows LangChain to handle run-time execution of chains in an optimized way.

While we have seen users run chains with hundreds of steps in production, we generally recommend using LCEL for simpler orchestration tasks. When the application requires complex state management, branching, cycles or multiple agents, we recommend that users take advantage of LangGraph.

In LangGraph, users define graphs that specify the flow of the application. This allows users to keep using LCEL within individual nodes when LCEL is needed, while making it easy to define complex orchestration logic that is more readable and maintainable.

Here are some guidelines:

If you are making a single LLM call, you don't need LCEL; instead call the underlying chat model directly.
If you have a simple chain (e.g., prompt + llm + parser, simple retrieval set up etc.), LCEL is a reasonable fit, if you're taking advantage of the LCEL benefits.
If you're building a complex chain (e.g., with branching, cycles, multiple agents, etc.) use LangGraph instead. Remember that you can always use LCEL within individual nodes in LangGraph.
Composition Primitives
LCEL chains are built by composing existing Runnables together. The two main composition primitives are RunnableSequence and RunnableParallel.

Many other composition primitives (e.g., RunnableAssign) can be thought of as variations of these two primitives.

note
You can find a list of all composition primitives in the LangChain Core API Reference.

RunnableSequence
RunnableSequence is a composition primitive that allows you "chain" multiple runnables sequentially, with the output of one runnable serving as the input to the next.

import { RunnableSequence } from "@langchain/core/runnables";
const chain = new RunnableSequence({
  first: runnable1,
  // Optional, use if you have more than two runnables
  // middle: [...],
  last: runnable2,
});

Invoking the chain with some input:

const finalOutput = await chain.invoke(someInput);

corresponds to the following:

const output1 = await runnable1.invoke(someInput);
const finalOutput = await runnable2.invoke(output1);

note
runnable1 and runnable2 are placeholders for any Runnable that you want to chain together.

RunnableParallel
RunnableParallel is a composition primitive that allows you to run multiple runnables concurrently, with the same input provided to each.

import { RunnableParallel } from "@langchain/core/runnables";
const chain = new RunnableParallel({
  key1: runnable1,
  key2: runnable2,
});

Invoking the chain with some input:

const finalOutput = await chain.invoke(someInput);

Will yield a finalOutput object with the same keys as the input object, but with the values replaced by the output of the corresponding runnable.

{
  key1: await runnable1.invoke(someInput),
  key2: await runnable2.invoke(someInput),
}

Recall, that the runnables are executed in parallel, so while the result is the same as object comprehension shown above, the execution time is much faster.

Composition Syntax
The usage of RunnableSequence and RunnableParallel is so common that we created a shorthand syntax for using them. This helps to make the code more readable and concise.

The pipe method.
You can pipe runnables together using the .pipe(runnable) method.

const chain = runnable1.pipe(runnable2);

is Equivalent to:

const chain = new RunnableSequence({
  first: runnable1,
  last: runnable2,
});

RunnableLambda functions
You can define generic TypeScript functions are runnables through the RunnableLambda class.

const someFunc = RunnableLambda.from((input) => {
  return input;
});

const chain = someFunc.pipe(runnable1);

Legacy chains
LCEL aims to provide consistency around behavior and customization over legacy subclassed chains such as LLMChain and ConversationalRetrievalChain. Many of these legacy chains hide important details like prompts, and as a wider variety of viable models emerge, customization has become more and more important.

For guides on how to do specific tasks with LCEL, check out the relevant how-to guides.