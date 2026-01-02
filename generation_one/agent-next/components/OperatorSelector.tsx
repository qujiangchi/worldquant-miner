'use client';

import { useState, useEffect } from 'react';
import { getStoredJWT } from '../lib/auth';

interface Operator {
  name: string;
  category: string;
  definition: string;
  description: string;
  documentation: string | null;
  level: string;
}

interface OperatorSelectorProps {
  onOperatorsSelected: (operators: string[]) => void;
  selectedOperators?: string[];
}

export default function OperatorSelector({ onOperatorsSelected, selectedOperators = [] }: OperatorSelectorProps) {
  const [operators, setOperators] = useState<Operator[]>([]);
  const [internalSelectedOperators, setInternalSelectedOperators] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  useEffect(() => {
    setInternalSelectedOperators(selectedOperators);
  }, [selectedOperators]);

  useEffect(() => {
    const fetchOperators = async () => {
      try {
        // Get the JWT token
        const jwtToken = getStoredJWT();
        
        if (!jwtToken) {
          setError('Authentication required. Please log in again.');
          setIsLoading(false);
          return;
        }
        
        const response = await fetch('/api/operators', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            jwtToken,
          }),
        });
        
        if (!response.ok) {
          throw new Error('Failed to fetch operators');
        }
        
        const data = await response.json();
        setOperators(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch operators');
      } finally {
        setIsLoading(false);
      }
    };

    fetchOperators();
  }, []);

  const handleOperatorToggle = (operator: string) => {
    const newSelectedOperators = internalSelectedOperators.includes(operator)
      ? internalSelectedOperators.filter(o => o !== operator)
      : [...internalSelectedOperators, operator];
    
    setInternalSelectedOperators(newSelectedOperators);
    onOperatorsSelected(newSelectedOperators);
  };

  // Get unique categories
  const categories = ['all', ...new Set(operators.map(op => op.category))];

  // Filter operators based on search term and selected category
  const filteredOperators = operators.filter(operator => {
    const matchesSearch = operator.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                         operator.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || operator.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20">
      <h2 className="text-xl font-semibold mb-4">Select Operators</h2>
      
      <div className="mb-4">
        <input
          type="text"
          placeholder="Search operators..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-md shadow-sm text-white placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
        />
      </div>
      
      <div className="mb-4">
        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-md shadow-sm text-white focus:outline-none focus:ring-blue-500 focus:border-blue-500"
        >
          {categories.map((category) => (
            <option key={category} value={category}>
              {category === 'all' ? 'All Categories' : category}
            </option>
          ))}
        </select>
      </div>
      
      {isLoading ? (
        <div className="flex justify-center py-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      ) : error ? (
        <div className="text-red-400 p-4 bg-red-900/30 rounded-lg">
          {error}
        </div>
      ) : (
        <div className="max-h-60 overflow-y-auto space-y-2">
          {filteredOperators.map((operator) => (
            <div 
              key={operator.name}
              className="flex items-start p-2 rounded-md hover:bg-white/10 transition-colors"
            >
              <input
                type="checkbox"
                checked={internalSelectedOperators.includes(operator.name)}
                onChange={() => handleOperatorToggle(operator.name)}
                className="mt-1 h-4 w-4 text-blue-400 rounded border-white/20 bg-white/10"
              />
              <div className="ml-3">
                <div className="font-medium text-white">{operator.name}</div>
                <div className="text-sm text-gray-300">{operator.description}</div>
              </div>
            </div>
          ))}
        </div>
      )}
      
      {internalSelectedOperators.length > 0 && (
        <div className="mt-4 p-3 bg-blue-900/30 rounded-lg">
          <p className="text-sm text-blue-200">
            {internalSelectedOperators.length} operator{internalSelectedOperators.length !== 1 ? 's' : ''} selected
          </p>
        </div>
      )}
    </div>
  );
} 