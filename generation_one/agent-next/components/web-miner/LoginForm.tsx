'use client';

import { useState } from 'react';
import { motion } from 'motion/react';
import { IconUser, IconLock, IconAlertCircle } from '@tabler/icons-react';
import { authenticateWorldQuant, storeCredentials } from '@/lib/auth';

interface LoginFormProps {
  onLoginSuccess: () => void;
}

export function LoginForm({ onLoginSuccess }: LoginFormProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!username || !password) {
      setError('Please enter both username and password');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Authenticate with WorldQuant API
      const credentials = await authenticateWorldQuant({
        username,
        password,
        token: '' // Will be filled by the authentication process
      });
      
      // Store credentials for future use
      storeCredentials(credentials);
      
      // Notify parent component of successful login
      onLoginSuccess();
    } catch (error) {
      console.error('Login error:', error);
      setError(error instanceof Error ? error.message : 'Authentication failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div
      className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20 max-w-md mx-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-xl font-bold mb-6 text-center">WorldQuant Brain Login</h2>
      
      {error && (
        <motion.div
          className="p-3 mb-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 flex items-start gap-2"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <IconAlertCircle className="h-5 w-5 mt-0.5 flex-shrink-0" />
          <p>{error}</p>
        </motion.div>
      )}
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="username" className="block text-sm font-medium mb-1">
            Username
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <IconUser className="h-5 w-5 text-blue-300" />
            </div>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="backdrop-blur-md bg-white/10 p-3 pl-10 rounded-xl border border-white/20 w-full focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              placeholder="Enter your WorldQuant username"
              disabled={isLoading}
            />
          </div>
        </div>
        
        <div>
          <label htmlFor="password" className="block text-sm font-medium mb-1">
            Password
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <IconLock className="h-5 w-5 text-blue-300" />
            </div>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="backdrop-blur-md bg-white/10 p-3 pl-10 rounded-xl border border-white/20 w-full focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              placeholder="Enter your WorldQuant password"
              disabled={isLoading}
            />
          </div>
        </div>
        
        <button
          type="submit"
          disabled={isLoading}
          className={`w-full p-3 rounded-xl transition-all ${
            isLoading
              ? 'bg-blue-500/30 text-blue-200 cursor-not-allowed'
              : 'bg-blue-500/50 text-white hover:bg-blue-500/70'
          }`}
        >
          {isLoading ? (
            <div className="flex justify-center items-center">
              <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full mr-2"></div>
              Authenticating...
            </div>
          ) : (
            'Login to WorldQuant Brain'
          )}
        </button>
      </form>
      
      <p className="mt-4 text-sm text-blue-200 text-center">
        Your credentials are stored locally and never sent to our servers.
      </p>
    </motion.div>
  );
} 