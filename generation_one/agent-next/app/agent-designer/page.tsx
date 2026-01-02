"use client"

import { useState } from "react"
import Link from "next/link"
import { Brain, Save, Play, Code, Settings, ArrowLeft, Plus, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"

export default function AgentDesignerPage() {
  const [agentName, setAgentName] = useState("New Alpha Mining Agent")
  const [agentDescription, setAgentDescription] = useState("")
  const [agentType, setAgentType] = useState("analyzer")
  const [agentCode, setAgentCode] = useState(`// Alpha mining agent code
// This is where you define the agent's behavior

function analyzeMarket(data) {
  // Example analysis logic
  const trends = data.filter(item => item.close > item.open);
  const sentiment = calculateSentiment(data);
  
  return {
    trends,
    sentiment,
    recommendations: generateRecommendations(trends, sentiment)
  };
}

function calculateSentiment(data) {
  // Example sentiment calculation
  return data.reduce((acc, item) => acc + item.sentiment, 0) / data.length;
}

function generateRecommendations(trends, sentiment) {
  // Example recommendation logic
  if (trends.length > 5 && sentiment > 0.6) {
    return [
      {
        type: "alpha_expression",
        expression: "close_5d_ret > 0.05 and volume_10d_avg > 1000000 and rsi_14 < 30",
        confidence: 0.82
      }
    ];
  }
  return [];
}

// Export the agent's main function
export default analyzeMarket;`)

  const [parameters, setParameters] = useState([
    {
      id: "1",
      name: "lookbackPeriod",
      type: "number",
      value: "14",
      description: "Number of days to look back for analysis",
    },
    {
      id: "2",
      name: "sentimentThreshold",
      type: "number",
      value: "0.6",
      description: "Threshold for positive sentiment",
    },
    {
      id: "3",
      name: "useMarketData",
      type: "boolean",
      value: "true",
      description: "Whether to use market data in analysis",
    },
  ])

  const addParameter = () => {
    const newId = (parameters.length + 1).toString()
    setParameters([...parameters, { id: newId, name: "", type: "string", value: "", description: "" }])
  }

  const removeParameter = (id: string) => {
    setParameters(parameters.filter((param) => param.id !== id))
  }

  const updateParameter = (id: string, field: string, value: string) => {
    setParameters(parameters.map((param) => (param.id === id ? { ...param, [field]: value } : param)))
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
            <Link href="/agents">
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          <h1 className="text-3xl font-bold">{agentName || "New Agent"}</h1>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Agent Details</CardTitle>
                <CardDescription>Configure the basic details of your AI agent.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="agent-name">Agent Name</Label>
                  <Input
                    id="agent-name"
                    value={agentName}
                    onChange={(e) => setAgentName(e.target.value)}
                    placeholder="Enter agent name"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="agent-description">Description</Label>
                  <Textarea
                    id="agent-description"
                    value={agentDescription}
                    onChange={(e) => setAgentDescription(e.target.value)}
                    placeholder="Describe what this agent does"
                    rows={4}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="agent-type">Agent Type</Label>
                  <Select value={agentType} onValueChange={setAgentType}>
                    <SelectTrigger id="agent-type">
                      <SelectValue placeholder="Select agent type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="analyzer">Analyzer</SelectItem>
                      <SelectItem value="evaluator">Evaluator</SelectItem>
                      <SelectItem value="recognizer">Recognizer</SelectItem>
                      <SelectItem value="predictor">Predictor</SelectItem>
                      <SelectItem value="finder">Finder</SelectItem>
                      <SelectItem value="generator">Generator</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch id="active" defaultChecked />
                  <Label htmlFor="active">Active</Label>
                </div>
              </CardContent>
              <CardFooter>
                <Button className="w-full gap-1">
                  <Save className="h-4 w-4" /> Save Agent
                </Button>
              </CardFooter>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Parameters</CardTitle>
                <CardDescription>Configure the parameters for this agent.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {parameters.map((param) => (
                  <div key={param.id} className="space-y-2 border-b pb-4 last:border-0">
                    <div className="flex justify-between items-center">
                      <Label htmlFor={`param-name-${param.id}`}>Parameter Name</Label>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-muted-foreground"
                        onClick={() => removeParameter(param.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                    <Input
                      id={`param-name-${param.id}`}
                      value={param.name}
                      onChange={(e) => updateParameter(param.id, "name", e.target.value)}
                      placeholder="Parameter name"
                    />
                    <div className="grid grid-cols-2 gap-2">
                      <div className="space-y-2">
                        <Label htmlFor={`param-type-${param.id}`}>Type</Label>
                        <Select value={param.type} onValueChange={(value) => updateParameter(param.id, "type", value)}>
                          <SelectTrigger id={`param-type-${param.id}`}>
                            <SelectValue placeholder="Type" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="string">String</SelectItem>
                            <SelectItem value="number">Number</SelectItem>
                            <SelectItem value="boolean">Boolean</SelectItem>
                            <SelectItem value="array">Array</SelectItem>
                            <SelectItem value="object">Object</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor={`param-value-${param.id}`}>Default Value</Label>
                        <Input
                          id={`param-value-${param.id}`}
                          value={param.value}
                          onChange={(e) => updateParameter(param.id, "value", e.target.value)}
                          placeholder="Default value"
                        />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor={`param-desc-${param.id}`}>Description</Label>
                      <Input
                        id={`param-desc-${param.id}`}
                        value={param.description}
                        onChange={(e) => updateParameter(param.id, "description", e.target.value)}
                        placeholder="Parameter description"
                      />
                    </div>
                  </div>
                ))}
                <Button variant="outline" className="w-full gap-1" onClick={addParameter}>
                  <Plus className="h-4 w-4" /> Add Parameter
                </Button>
              </CardContent>
            </Card>
          </div>
          <div className="lg:col-span-2">
            <Card className="h-full flex flex-col">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Agent Implementation</CardTitle>
                  <Tabs defaultValue="code" className="w-[400px]">
                    <TabsList>
                      <TabsTrigger value="code" className="flex items-center gap-1">
                        <Code className="h-4 w-4" /> Code
                      </TabsTrigger>
                      <TabsTrigger value="settings" className="flex items-center gap-1">
                        <Settings className="h-4 w-4" /> Settings
                      </TabsTrigger>
                      <TabsTrigger value="test" className="flex items-center gap-1">
                        <Play className="h-4 w-4" /> Test
                      </TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>
                <CardDescription>
                  Write the code that defines how this agent processes data and generates alpha expressions.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex-1 p-0">
                <div className="border rounded-md h-[600px] font-mono text-sm p-4 bg-muted overflow-auto">
                  <pre className="whitespace-pre-wrap">
                    <textarea
                      className="w-full h-full bg-transparent outline-none resize-none"
                      value={agentCode}
                      onChange={(e) => setAgentCode(e.target.value)}
                    />
                  </pre>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline" className="gap-1">
                  <Play className="h-4 w-4" /> Test Agent
                </Button>
                <Button className="gap-1">
                  <Save className="h-4 w-4" /> Save Changes
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

