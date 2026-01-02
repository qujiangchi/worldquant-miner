import { type NextRequest, NextResponse } from "next/server"
import { runAgentNetwork, type AgentConfig } from "@/lib/agent-service"

export async function POST(request: NextRequest) {
  try {
    const { agents, data } = await request.json()

    if (!agents || !data || !Array.isArray(agents)) {
      return NextResponse.json({ error: "Missing required parameters or invalid format" }, { status: 400 })
    }

    const result = await runAgentNetwork(agents as AgentConfig[], data)

    return NextResponse.json({ expressions: result })
  } catch (error) {
    console.error("Error in run-network API route:", error)
    return NextResponse.json({ error: "Failed to run agent network" }, { status: 500 })
  }
}

