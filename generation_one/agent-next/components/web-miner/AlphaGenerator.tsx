'use client';

import { useState } from 'react';
import { motion } from 'motion/react';
import { IconBrain, IconRefresh, IconPlayerPlay, IconPlayerStop, IconDownload, IconCopy, IconCheck } from '@tabler/icons-react';

interface AlphaIdea {
  id: string;
  name: string;
  description: string;
  formula: string;
  dataFields: string[];
  performance: {
    sharpe: number;
    returns: number;
    drawdown: number;
  };
}

interface Operator {
  id: string;
  name: string;
  description: string;
  category: string;
}

interface AlphaGeneratorProps {
  selectedFields: string[];
  operators: Operator[];
  onGenerate: () => void;
}

export function AlphaGenerator({ selectedFields, operators, onGenerate }: AlphaGeneratorProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedIdeas, setGeneratedIdeas] = useState<AlphaIdea[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (selectedFields.length === 0) {
      alert('Please select at least one data field');
      return;
    }

    setIsGenerating(true);
    setError(null);
    
    try {
      // Call the API to generate alpha ideas
      const response = await fetch('/api/worldquant', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fields: selectedFields,
          operators: operators.map(op => op.id),
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to generate alpha ideas');
      }
      
      const data = await response.json();
      
      if (data.ideas && Array.isArray(data.ideas)) {
        setGeneratedIdeas(data.ideas);
        onGenerate();
      } else {
        throw new Error('Invalid response format from API');
      }
    } catch (error) {
      console.error('Error generating alpha ideas:', error);
      setError(error instanceof Error ? error.message : 'Failed to generate alpha ideas');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCopyFormula = async (formula: string, id: string) => {
    try {
      await navigator.clipboard.writeText(formula);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (error) {
      console.error('Failed to copy formula:', error);
    }
  };

  const handleDownload = (idea: AlphaIdea) => {
    const data = {
      name: idea.name,
      description: idea.description,
      formula: idea.formula,
      dataFields: idea.dataFields,
      performance: idea.performance
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${idea.name.toLowerCase().replace(/\s+/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <motion.div
      className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-lg font-bold">Alpha Ideas</h3>
        <button
          onClick={handleGenerate}
          disabled={isGenerating || selectedFields.length === 0}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            isGenerating || selectedFields.length === 0
              ? 'bg-white/5 text-blue-200/50 cursor-not-allowed'
              : 'bg-blue-500/20 text-blue-300 hover:bg-blue-500/30'
          }`}
        >
          {isGenerating ? (
            <>
              <div className="animate-spin h-4 w-4 border-2 border-blue-300 border-t-transparent rounded-full"></div>
              Generating...
            </>
          ) : (
            <>
              <IconBrain className="h-5 w-5" />
              Generate Ideas
            </>
          )}
        </button>
      </div>
      
      {error && (
        <div className="p-4 mb-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400">
          <p className="font-medium">Error generating alpha ideas</p>
          <p className="text-sm">{error}</p>
        </div>
      )}
      
      {generatedIdeas.length > 0 ? (
        <div className="space-y-4">
          {generatedIdeas.map((idea) => (
            <motion.div
              key={idea.id}
              className="p-4 rounded-lg border border-white/20 bg-white/5"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="flex justify-between items-start mb-3">
                <div>
                  <h4 className="font-medium">{idea.name}</h4>
                  <p className="text-sm text-blue-200 mt-1">{idea.description}</p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleCopyFormula(idea.formula, idea.id)}
                    className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                    title="Copy Formula"
                  >
                    {copiedId === idea.id ? (
                      <IconCheck className="h-4 w-4 text-green-400" />
                    ) : (
                      <IconCopy className="h-4 w-4" />
                    )}
                  </button>
                  <button
                    onClick={() => handleDownload(idea)}
                    className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                    title="Download"
                  >
                    <IconDownload className="h-4 w-4" />
                  </button>
                </div>
              </div>
              
              <div className="mb-3">
                <div className="text-sm font-medium mb-1">Formula</div>
                <code className="block p-2 rounded bg-white/5 text-sm font-mono">
                  {idea.formula}
                </code>
              </div>
              
              <div className="grid grid-cols-3 gap-4">
                <div className="p-2 rounded bg-white/5">
                  <div className="text-xs text-blue-200">Sharpe Ratio</div>
                  <div className="text-lg font-medium">{idea.performance.sharpe.toFixed(2)}</div>
                </div>
                <div className="p-2 rounded bg-white/5">
                  <div className="text-xs text-blue-200">Returns (%)</div>
                  <div className="text-lg font-medium">{idea.performance.returns.toFixed(1)}%</div>
                </div>
                <div className="p-2 rounded bg-white/5">
                  <div className="text-xs text-blue-200">Max Drawdown</div>
                  <div className="text-lg font-medium">{idea.performance.drawdown.toFixed(1)}%</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-blue-200">
          {selectedFields.length === 0 ? (
            <p>Select data fields to generate alpha ideas</p>
          ) : (
            <p>Click "Generate Ideas" to create alpha strategies</p>
          )}
        </div>
      )}
    </motion.div>
  );
} 