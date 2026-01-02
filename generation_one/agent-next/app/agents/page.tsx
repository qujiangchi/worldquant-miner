import Link from "next/link"
import { Brain, Plus, Search, Settings } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"

export default function AgentsPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <header className="border-b">
        <div className="container flex h-16 items-center justify-between px-4 md:px-6">
          <div className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-primary" />
            <span className="text-xl font-bold">AlphaMind</span>
          </div>
          <nav className="hidden md:flex gap-6">
            <Link href="/" className="text-sm font-medium hover:underline underline-offset-4">
              Home
            </Link>
            <Link href="/agents" className="text-sm font-medium hover:underline underline-offset-4 text-primary">
              Agents
            </Link>
            <Link href="/networks" className="text-sm font-medium hover:underline underline-offset-4">
              Networks
            </Link>
            <Link href="/results" className="text-sm font-medium hover:underline underline-offset-4">
              Results
            </Link>
          </nav>
          <div className="flex items-center gap-4">
            <Button variant="outline" size="sm">
              Sign In
            </Button>
            <Button size="sm">Get Started</Button>
          </div>
        </div>
      </header>
      <main className="flex-1 container px-4 py-6 md:px-6 md:py-8">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">AI Agents</h1>
          <Button className="gap-1">
            <Plus className="h-4 w-4" /> Create Agent
          </Button>
        </div>
        <div className="flex flex-col gap-6">
          <div className="flex flex-col md:flex-row gap-4 justify-between">
            <div className="relative w-full md:w-96">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input type="search" placeholder="Search agents..." className="w-full pl-8" />
            </div>
            <Tabs defaultValue="all" className="w-full md:w-auto">
              <TabsList>
                <TabsTrigger value="all">All Agents</TabsTrigger>
                <TabsTrigger value="active">Active</TabsTrigger>
                <TabsTrigger value="draft">Draft</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {agents.map((agent) => (
              <AgentCard key={agent.id} agent={agent} />
            ))}
          </div>
        </div>
      </main>
      <footer className="border-t">
        <div className="container flex flex-col gap-4 py-10 md:flex-row md:gap-8 md:py-12">
          <div className="flex flex-col gap-2 md:gap-4">
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              <span className="text-lg font-bold">AlphaMind</span>
            </div>
            <p className="text-sm text-muted-foreground">AI-powered financial alpha mining platform</p>
          </div>
          <div className="ml-auto flex flex-col gap-2 md:flex-row md:gap-8">
            <div className="flex flex-col gap-2">
              <h3 className="text-sm font-medium">Platform</h3>
              <nav className="flex flex-col gap-2">
                <Link href="#" className="text-sm text-muted-foreground hover:underline">
                  Features
                </Link>
                <Link href="#" className="text-sm text-muted-foreground hover:underline">
                  Pricing
                </Link>
                <Link href="#" className="text-sm text-muted-foreground hover:underline">
                  Documentation
                </Link>
              </nav>
            </div>
            <div className="flex flex-col gap-2">
              <h3 className="text-sm font-medium">Company</h3>
              <nav className="flex flex-col gap-2">
                <Link href="#" className="text-sm text-muted-foreground hover:underline">
                  About
                </Link>
                <Link href="#" className="text-sm text-muted-foreground hover:underline">
                  Blog
                </Link>
                <Link href="#" className="text-sm text-muted-foreground hover:underline">
                  Contact
                </Link>
              </nav>
            </div>
            <div className="flex flex-col gap-2">
              <h3 className="text-sm font-medium">Legal</h3>
              <nav className="flex flex-col gap-2">
                <Link href="#" className="text-sm text-muted-foreground hover:underline">
                  Terms
                </Link>
                <Link href="#" className="text-sm text-muted-foreground hover:underline">
                  Privacy
                </Link>
              </nav>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

interface Agent {
  id: string
  name: string
  description: string
  type: string
  status: "active" | "draft"
  lastUpdated: string
}

function AgentCard({ agent }: { agent: Agent }) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{agent.name}</CardTitle>
          <Badge variant={agent.status === "active" ? "default" : "secondary"}>{agent.status}</Badge>
        </div>
        <CardDescription>{agent.description}</CardDescription>
      </CardHeader>
      <CardContent className="pb-3">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>Type: {agent.type}</span>
          <span>•</span>
          <span>Last updated: {agent.lastUpdated}</span>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" size="sm">
          Edit
        </Button>
        <Button variant="outline" size="sm" className="gap-1">
          <Settings className="h-4 w-4" /> Configure
        </Button>
      </CardFooter>
    </Card>
  )
}

const agents: Agent[] = [
  {
    id: "1",
    name: "Market Trend Analyzer",
    description: "Analyzes market trends and identifies potential opportunities.",
    type: "Analyzer",
    status: "active",
    lastUpdated: "2 days ago",
  },
  {
    id: "2",
    name: "Sentiment Evaluator",
    description: "Evaluates market sentiment from news and social media.",
    type: "Evaluator",
    status: "active",
    lastUpdated: "1 week ago",
  },
  {
    id: "3",
    name: "Pattern Recognition",
    description: "Identifies patterns in historical price data.",
    type: "Recognizer",
    status: "active",
    lastUpdated: "3 days ago",
  },
  {
    id: "4",
    name: "Volatility Predictor",
    description: "Predicts market volatility based on historical data.",
    type: "Predictor",
    status: "draft",
    lastUpdated: "5 days ago",
  },
  {
    id: "5",
    name: "Correlation Finder",
    description: "Finds correlations between different assets and markets.",
    type: "Finder",
    status: "active",
    lastUpdated: "1 day ago",
  },
  {
    id: "6",
    name: "Alpha Expression Generator",
    description: "Generates alpha expressions based on market data.",
    type: "Generator",
    status: "draft",
    lastUpdated: "4 days ago",
  },
]

