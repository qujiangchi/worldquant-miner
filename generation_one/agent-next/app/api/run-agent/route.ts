import { type NextRequest, NextResponse } from "next/server"
import { runAgent, type AgentConfig } from "@/lib/agent-service"

export async function POST(request: NextRequest) {
  try {
    const { agent, data } = await request.json()

    if (!agent || !data) {
      return NextResponse.json({ error: "Missing required parameters" }, { status: 400 })
    }

    const result = await runAgent(agent as AgentConfig, data)

    return NextResponse.json(result)
  } catch (error) {
    console.error("Error in run-agent API route:", error)
    return NextResponse.json({ error: "Failed to run agent" }, { status: 500 })
  }
}

