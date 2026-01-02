"use client"

import { useState } from "react"
import Link from "next/link"
import { Brain, Save, Play, ArrowLeft, Plus, Trash2, Network, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"

export default function NetworkDesignerPage() {
  const [networkName, setNetworkName] = useState("New Agent Network")
  const [networkDescription, setNetworkDescription] = useState("")
  const [selectedAgents, setSelectedAgents] = useState<string[]>(["1", "2", "5"])

  const addAgent = (id: string) => {
    if (!selectedAgents.includes(id)) {
      setSelectedAgents([...selectedAgents, id])
    }
  }

  const removeAgent = (id: string) => {
    setSelectedAgents(selectedAgents.filter((agentId) => agentId !== id))
  }

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
        <div className="flex items-center gap-2 mb-8">
          <Button variant="outline" size="icon" asChild>
            <Link href="/networks">
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          <h1 className="text-3xl font-bold">{networkName || "New Network"}</h1>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Network Details</CardTitle>
                <CardDescription>Configure the basic details of your agent network.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="network-name">Network Name</Label>
                  <Input
                    id="network-name"
                    value={networkName}
                    onChange={(e) => setNetworkName(e.target.value)}
                    placeholder="Enter network name"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="network-description">Description</Label>
                  <Textarea
                    id="network-description"
                    value={networkDescription}
                    onChange={(e) => setNetworkDescription(e.target.value)}
                    placeholder="Describe what this network does"
                    rows={4}
                  />
                </div>
                <div className="flex items-center space-x-2">
                  <Switch id="active" defaultChecked />
                  <Label htmlFor="active">Active</Label>
                </div>
              </CardContent>
              <CardFooter>
                <Button className="w-full gap-1">
                  <Save className="h-4 w-4" /> Save Network
                </Button>
              </CardFooter>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Network Settings</CardTitle>
                <CardDescription>Configure how this network operates.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="execution-mode">Execution Mode</Label>
                  <div className="flex items-center space-x-2">
                    <Switch id="parallel" defaultChecked />
                    <Label htmlFor="parallel">Parallel Execution</Label>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="data-source">Data Source</Label>
                  <div className="flex items-center space-x-2">
                    <Switch id="real-time" defaultChecked />
                    <Label htmlFor="real-time">Real-time Data</Label>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="auto-optimize">Auto-Optimization</Label>
                  <div className="flex items-center space-x-2">
                    <Switch id="auto-optimize" defaultChecked />
                    <Label htmlFor="auto-optimize">Enable Auto-Optimization</Label>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
          <div className="lg:col-span-2">
            <Card className="h-full flex flex-col">
              <CardHeader>
                <CardTitle>Network Design</CardTitle>
                <CardDescription>Select and connect agents to create your network.</CardDescription>
              </CardHeader>
              <CardContent className="flex-1">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-medium mb-4">Available Agents</h3>
                    <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
                      {availableAgents.map((agent) => (
                        <Card
                          key={agent.id}
                          className={`border ${selectedAgents.includes(agent.id) ? "border-primary/50 bg-primary/5" : ""}`}
                        >
                          <CardHeader className="p-4 pb-2">
                            <CardTitle className="text-base">{agent.name}</CardTitle>
                          </CardHeader>
                          <CardContent className="p-4 pt-0 pb-2">
                            <p className="text-sm text-muted-foreground">{agent.description}</p>
                          </CardContent>
                          <CardFooter className="p-4 pt-0 flex justify-between">
                            <Badge variant="outline">{agent.type}</Badge>
                            {selectedAgents.includes(agent.id) ? (
                              <Button
                                variant="outline"
                                size="sm"
                                className="gap-1"
                                onClick={() => removeAgent(agent.id)}
                              >
                                <Trash2 className="h-3 w-3" /> Remove
                              </Button>
                            ) : (
                              <Button variant="outline" size="sm" className="gap-1" onClick={() => addAgent(agent.id)}>
                                <Plus className="h-3 w-3" /> Add
                              </Button>
                            )}
                          </CardFooter>
                        </Card>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h3 className="text-lg font-medium mb-4">Network Structure</h3>
                    <div className="border rounded-md p-4 h-[500px] bg-muted/50 relative">
                      <div className="flex items-center justify-center h-full">
                        {selectedAgents.length > 0 ? (
                          <div className="flex flex-col items-center">
                            <Network className="h-16 w-16 text-primary/40 mb-4" />
                            <div className="space-y-2 w-full">
                              {selectedAgents.map((agentId) => {
                                const agent = availableAgents.find((a) => a.id === agentId)
                                return agent ? (
                                  <div
                                    key={agent.id}
                                    className="flex items-center gap-2 bg-background p-2 rounded-md border"
                                  >
                                    <Badge variant="outline" className="w-24 justify-center">
                                      {agent.type}
                                    </Badge>
                                    <span className="text-sm font-medium">{agent.name}</span>
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      className="h-6 w-6 ml-auto"
                                      onClick={() => removeAgent(agent.id)}
                                    >
                                      <Trash2 className="h-3 w-3" />
                                    </Button>
                                  </div>
                                ) : null
                              })}
                            </div>
                            <div className="mt-4 flex flex-col items-center">
                              <ArrowRight className="h-6 w-6 text-primary/60 rotate-90 mb-2" />
                              <div className="bg-background p-2 rounded-md border">
                                <span className="text-sm font-medium">Alpha Expressions</span>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="text-center text-muted-foreground">
                            <Network className="h-16 w-16 mx-auto mb-4 text-muted-foreground/40" />
                            <p>Add agents to build your network</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline" className="gap-1">
                  <Play className="h-4 w-4" /> Test Network
                </Button>
                <Button className="gap-1">
                  <Save className="h-4 w-4" /> Save Network
                </Button>
              </CardFooter>
            </Card>
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

const availableAgents = [
  {
    id: "1",
    name: "Market Trend Analyzer",
    description: "Analyzes market trends and identifies potential opportunities.",
    type: "Analyzer",
  },
  {
    id: "2",
    name: "Sentiment Evaluator",
    description: "Evaluates market sentiment from news and social media.",
    type: "Evaluator",
  },
  {
    id: "3",
    name: "Pattern Recognition",
    description: "Identifies patterns in historical price data.",
    type: "Recognizer",
  },
  {
    id: "4",
    name: "Volatility Predictor",
    description: "Predicts market volatility based on historical data.",
    type: "Predictor",
  },
  {
    id: "5",
    name: "Correlation Finder",
    description: "Finds correlations between different assets and markets.",
    type: "Finder",
  },
  {
    id: "6",
    name: "Alpha Expression Generator",
    description: "Generates alpha expressions based on market data.",
    type: "Generator",
  },
]

