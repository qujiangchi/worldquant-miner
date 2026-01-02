'use client';

import { useState, useEffect } from 'react';
import { getStoredJWT } from '../lib/auth';

interface DataField {
  id: string;
  name: string;
  description: string;
  dataset: {
    id: string;
    name: string;
  };
  category: {
    id: string;
    name: string;
  };
  subcategory?: {
    id: string;
    name: string;
  };
  region: string;
  delay: number;
  universe: string;
  type: string;
  coverage: number;
  userCount: number;
  alphaCount: number;
  themes: string[];
}

interface DataFieldSelectorProps {
  onFieldsSelected: (fields: string[]) => void;
  selectedFields?: string[];
  onDatasetChange?: (dataset: string) => void;
  onCategoryChange?: (category: string) => void;
  onPageChange?: (page: string) => void;
}

export default function DataFieldSelector({ 
  onFieldsSelected, 
  selectedFields = [],
  onDatasetChange,
  onCategoryChange,
  onPageChange
}: DataFieldSelectorProps) {
  const [fields, setFields] = useState<DataField[]>([]);
  const [internalSelectedFields, setInternalSelectedFields] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedDataset, setSelectedDataset] = useState<string>('all');
  const [dataset, setDataset] = useState('fundamental6');
  const [totalCount, setTotalCount] = useState(0);
  const [offset, setOffset] = useState('0');
  const [limit, setLimit] = useState('20');

  useEffect(() => {
    setInternalSelectedFields(selectedFields);
  }, [selectedFields]);

  useEffect(() => {
    const fetchFields = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        // Get the JWT token
        const jwtToken = getStoredJWT();
        
        if (!jwtToken) {
          setError('Authentication required. Please log in again.');
          setIsLoading(false);
          return;
        }
        
        // First get the count
        const countResponse = await fetch('/api/data-fields', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            jwtToken,
            dataset,
            limit: '1', // Just to get count efficiently
            instrumentType: 'EQUITY',
            region: 'USA',
            universe: 'TOP3000',
            delay: '1',
          }),
        });
        
        if (!countResponse.ok) {
          throw new Error('Failed to fetch data fields count');
        }
        
        const countData = await countResponse.json();
        const totalFields = countData.count || 0;
        setTotalCount(totalFields);
        
        if (totalFields > 0) {
          // Fetch a subset of fields
          const response = await fetch('/api/data-fields', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              jwtToken,
              dataset,
              limit,
              instrumentType: 'EQUITY',
              region: 'USA',
              universe: 'TOP3000',
              delay: '1',
              offset,
            }),
          });
          
          if (!response.ok) {
            throw new Error('Failed to fetch data fields');
          }
          
          const data = await response.json();
          setFields(data.results || []);
        } else {
          setFields([]);
        }
      } catch (err) {
        console.error('Error fetching data fields:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch data fields');
      } finally {
        setIsLoading(false);
      }
    };

    fetchFields();
  }, [dataset, offset, limit]);

  const handleFieldToggle = (field: DataField) => {
    const newSelectedFields = internalSelectedFields.includes(field.id)
      ? internalSelectedFields.filter(id => id !== field.id)
      : [...internalSelectedFields, field.id];
    
    setInternalSelectedFields(newSelectedFields);
    
    if (onFieldsSelected) {
      onFieldsSelected(newSelectedFields);
    }
  };

  // Get unique categories and datasets
  const categories = ['all', ...new Set(fields.map(field => field.category.name))];
  const datasets = ['all', ...new Set(fields.map(field => field.dataset.name))];

  // Filter fields based on search term, selected category, and selected dataset
  const filteredFields = fields.filter(field => {
    const matchesSearch = field.id.toLowerCase().includes(searchTerm.toLowerCase()) || 
                         field.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || field.category.name === selectedCategory;
    const matchesDataset = selectedDataset === 'all' || field.dataset.name === selectedDataset;
    return matchesSearch && matchesCategory && matchesDataset;
  });

  // Calculate pagination options
  const pageSize = parseInt(limit);
  const totalPages = Math.ceil(totalCount / pageSize);
  const currentPage = Math.floor(parseInt(offset) / pageSize) + 1;
  
  // Generate page options for dropdown
  const pageOptions = Array.from({ length: totalPages }, (_, i) => ({
    value: (i * pageSize).toString(),
    label: `Page ${i + 1} of ${totalPages}`
  }));

  // Update the select handlers to notify parent
  const handleDatasetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newDataset = e.target.value;
    setDataset(newDataset);
    onDatasetChange?.(newDataset);
  };

  const handleCategoryChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newCategory = e.target.value;
    setSelectedCategory(newCategory);
    onCategoryChange?.(newCategory);
  };

  const handlePageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newPage = e.target.value;
    setOffset(newPage);
    onPageChange?.(newPage);
  };

  return (
    <div className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20">
      <h2 className="text-xl font-semibold mb-4">Select Data Fields</h2>
      
      <div className="mb-4">
        <input
          type="text"
          placeholder="Search data fields..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-md shadow-sm text-white placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
        />
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div>
          <label htmlFor="dataset" className="block text-sm font-medium text-blue-200 mb-2">
            Dataset
          </label>
          <select
            id="dataset"
            value={dataset}
            onChange={handleDatasetChange}
            className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-md shadow-sm text-white focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="fundamental6" className="text-black">Fundamental 6</option>
            <option value="fundamental2" className="text-black">Fundamental 2</option>
            <option value="analyst4" className="text-black">Analyst 4</option>
            <option value="model16" className="text-black">Model 16</option>
            <option value="model51" className="text-black">Model 51</option>
            <option value="news12" className="text-black">News 12</option>
          </select>
        </div>
        
        <div>
          <label htmlFor="category" className="block text-sm font-medium text-blue-200 mb-2">
            Category
          </label>
          <select
            id="category"
            value={selectedCategory}
            onChange={(e) => {
              setSelectedCategory(e.target.value);
              onCategoryChange?.(e.target.value);
            }}
            className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-md shadow-sm text-white focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          >
            {categories.map(category => (
              <option key={category} value={category} className="text-black">
                {category.charAt(0).toUpperCase() + category.slice(1)}
              </option>
            ))}
          </select>
        </div>
        
        <div>
          <label htmlFor="page" className="block text-sm font-medium text-blue-200 mb-2">
            Page
          </label>
          <select
            id="page"
            value={offset}
            onChange={handlePageChange}
            className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-md shadow-sm text-white focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          >
            {pageOptions.map(option => (
              <option key={option.value} value={option.value} className="text-black">
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
        </div>
      ) : error ? (
        <div className="p-4 bg-red-900/50 text-red-200 rounded">
          {error}
        </div>
      ) : (
        <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
          {totalCount > 0 && (
            <div className="text-sm text-blue-300 mb-2">
              Showing {fields.length} of {totalCount} available fields (Page {currentPage} of {totalPages})
            </div>
          )}
          
          {filteredFields.map((field) => (
            <div
              key={field.id}
              className={`p-3 rounded-lg transition-colors ${
                internalSelectedFields.includes(field.id)
                  ? 'bg-blue-900/30 border border-blue-500'
                  : 'bg-white/5 hover:bg-white/10'
              }`}
            >
              <div className="flex items-start">
                <input
                  type="checkbox"
                  checked={internalSelectedFields.includes(field.id)}
                  onChange={() => handleFieldToggle(field)}
                  className="mt-1 h-4 w-4 text-blue-400 rounded border-white/20 bg-white/10"
                />
                <div className="ml-3 flex-1">
                  <div className="flex justify-between">
                    <h3 className="text-blue-200 font-medium">{field.id}</h3>
                    <div className="flex space-x-2">
                      <span className="text-xs px-2 py-1 bg-blue-900/30 text-blue-200 rounded-full">
                        {field.category.name}
                      </span>
                      <span className="text-xs px-2 py-1 bg-blue-900/30 text-blue-200 rounded-full">
                        {field.dataset.name}
                      </span>
                    </div>
                  </div>
                  <p className="text-sm text-blue-300 mt-1">{field.description}</p>
                  <div className="flex mt-1 text-xs text-blue-400 space-x-2">
                    <span>Region: {field.region}</span>
                    <span>Universe: {field.universe}</span>
                    <span>Delay: {field.delay}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
          
          {filteredFields.length === 0 && (
            <div className="text-center py-4 text-blue-300">
              No data fields found matching your criteria
            </div>
          )}
        </div>
      )}
      
      {internalSelectedFields.length > 0 && (
        <div className="mt-4 p-3 bg-blue-900/30 rounded-lg">
          <p className="text-sm text-blue-200">
            {internalSelectedFields.length} field{internalSelectedFields.length !== 1 ? 's' : ''} selected
          </p>
        </div>
      )}
    </div>
  );
} 