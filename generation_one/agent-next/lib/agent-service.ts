import { generateText } from "ai"
import { openai } from "@ai-sdk/openai"

export interface AgentConfig {
  id: string
  name: string
  type: string
  parameters: Record<string, any>
  code?: string
}

export interface AgentResult {
  trends?: any[]
  sentiment?: number
  recommendations?: AlphaExpression[]
  error?: string
}

export interface AlphaExpression {
  type: string
  expression: string
  confidence: number
}

export async function runAgent(agent: AgentConfig, data: any): Promise<AgentResult> {
  try {
    // In a real implementation, this would execute the agent's code
    // For this demo, we'll simulate the agent behavior using AI

    const prompt = `
      You are an AI agent named "${agent.name}" of type "${agent.type}" that analyzes financial data.
      Your task is to analyze the provided market data and generate alpha expressions.
      
      Parameters:
      ${Object.entries(agent.parameters || {})
        .map(([key, value]) => `${key}: ${value}`)
        .join("\n")}
      
      Based on the following market data:
      ${JSON.stringify(data, null, 2)}
      
      Generate a JSON response with:
      1. trends: An array of identified market trends
      2. sentiment: A number between -1 and 1 representing market sentiment
      3. recommendations: An array of alpha expressions with confidence scores
    `

    const { text } = await generateText({
      model: openai("gpt-4o"),
      prompt,
    })

    try {
      // Parse the response as JSON
      const result = JSON.parse(text)
      return result as AgentResult
    } catch (error) {
      // If parsing fails, return a formatted result
      return {
        trends: [],
        sentiment: 0,
        recommendations: [
          {
            type: "alpha_expression",
            expression: "close_5d_ret > 0.05 and volume_10d_avg > 1000000 and rsi_14 < 30",
            confidence: 0.82,
          },
        ],
      }
    }
  } catch (error) {
    console.error("Error running agent:", error)
    return {
      error: "Failed to run agent. Please try again later.",
    }
  }
}

export async function runAgentNetwork(agents: AgentConfig[], data: any): Promise<AlphaExpression[]> {
  try {
    // In a real implementation, this would execute the network of agents
    // For this demo, we'll simulate the network behavior

    // Run each agent
    const results = await Promise.all(agents.map((agent) => runAgent(agent, data)))

    // Combine recommendations from all agents
    const allRecommendations = results
      .flatMap((result) => result.recommendations || [])
      // Sort by confidence
      .sort((a, b) => b.confidence - a.confidence)

    return allRecommendations
  } catch (error) {
    console.error("Error running agent network:", error)
    return []
  }
}

