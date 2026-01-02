import Link from "next/link"
import { Brain, Network, Plus, Search, Settings, Play, Pause } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

export default function NetworksPage() {
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
            <Link href="/agents" className="text-sm font-medium hover:underline underline-offset-4">
              Agents
            </Link>
            <Link href="/networks" className="text-sm font-medium hover:underline underline-offset-4 text-primary">
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
          <h1 className="text-3xl font-bold">Agent Networks</h1>
          <Button className="gap-1">
            <Plus className="h-4 w-4" /> Create Network
          </Button>
        </div>
        <div className="flex flex-col gap-6">
          <div className="flex flex-col md:flex-row gap-4 justify-between">
            <div className="relative w-full md:w-96">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input type="search" placeholder="Search networks..." className="w-full pl-8" />
            </div>
            <Tabs defaultValue="all" className="w-full md:w-auto">
              <TabsList>
                <TabsTrigger value="all">All Networks</TabsTrigger>
                <TabsTrigger value="running">Running</TabsTrigger>
                <TabsTrigger value="paused">Paused</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {networks.map((network) => (
              <NetworkCard key={network.id} network={network} />
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

interface NetworkAgent {
  id: string
  name: string
  type: string
}

interface NetworkType {
  id: string
  name: string
  description: string
  status: "running" | "paused"
  progress: number
  agents: NetworkAgent[]
  alphaCount: number
  lastUpdated: string
}

function NetworkCard({ network }: { network: NetworkType }) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{network.name}</CardTitle>
          <Badge variant={network.status === "running" ? "default" : "secondary"}>{network.status}</Badge>
        </div>
        <CardDescription>{network.description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Progress</span>
            <span>{network.progress}%</span>
          </div>
          <Progress value={network.progress} />
        </div>
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Agents in Network</h4>
          <div className="flex flex-wrap gap-2">
            {network.agents.map((agent) => (
              <Badge key={agent.id} variant="outline">
                {agent.name}
              </Badge>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <Network className="h-4 w-4" />
            <span>{network.agents.length} Agents</span>
          </div>
          <span>•</span>
          <span>{network.alphaCount} Alpha Expressions</span>
          <span>•</span>
          <span>Updated {network.lastUpdated}</span>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" size="sm">
          View Details
        </Button>
        <div className="flex gap-2">
          {network.status === "running" ? (
            <Button variant="outline" size="sm" className="gap-1">
              <Pause className="h-4 w-4" /> Pause
            </Button>
          ) : (
            <Button variant="outline" size="sm" className="gap-1">
              <Play className="h-4 w-4" /> Run
            </Button>
          )}
          <Button variant="outline" size="sm" className="gap-1">
            <Settings className="h-4 w-4" /> Configure
          </Button>
        </div>
      </CardFooter>
    </Card>
  )
}

const networks: NetworkType[] = [
  {
    id: "1",
    name: "Market Trend Network",
    description: "Network focused on identifying market trends and generating alpha expressions.",
    status: "running",
    progress: 67,
    agents: [
      { id: "1", name: "Market Trend Analyzer", type: "Analyzer" },
      { id: "2", name: "Sentiment Evaluator", type: "Evaluator" },
      { id: "3", name: "Pattern Recognition", type: "Recognizer" },
    ],
    alphaCount: 12,
    lastUpdated: "2 hours ago",
  },
  {
    id: "2",
    name: "Volatility Prediction Network",
    description: "Network designed to predict market volatility and generate alpha expressions.",
    status: "paused",
    progress: 45,
    agents: [
      { id: "4", name: "Volatility Predictor", type: "Predictor" },
      { id: "5", name: "Correlation Finder", type: "Finder" },
    ],
    alphaCount: 8,
    lastUpdated: "1 day ago",
  },
  {
    id: "3",
    name: "Sector Rotation Network",
    description: "Network analyzing sector rotation patterns and generating alpha expressions.",
    status: "running",
    progress: 89,
    agents: [
      { id: "1", name: "Market Trend Analyzer", type: "Analyzer" },
      { id: "5", name: "Correlation Finder", type: "Finder" },
      { id: "6", name: "Alpha Expression Generator", type: "Generator" },
    ],
    alphaCount: 15,
    lastUpdated: "5 hours ago",
  },
  {
    id: "4",
    name: "Global Macro Network",
    description: "Network analyzing global macroeconomic factors and generating alpha expressions.",
    status: "running",
    progress: 32,
    agents: [
      { id: "2", name: "Sentiment Evaluator", type: "Evaluator" },
      { id: "5", name: "Correlation Finder", type: "Finder" },
      { id: "6", name: "Alpha Expression Generator", type: "Generator" },
    ],
    alphaCount: 6,
    lastUpdated: "12 hours ago",
  },
]

