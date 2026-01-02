'use client';

import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { useRouter } from 'next/navigation';
import { FloatingDock } from '@/components/ui/floating-dock';
import { sharedNavItems } from '@/components/ui/shared-navigation';
import { 
  IconHome, 
  IconChartBar, 
  IconBrain, 
  IconSettings, 
  IconUser,
  IconLogout,
  IconArrowUpRight,
  IconArrowDownRight,
  IconClock,
  IconActivity,
  IconWorld
} from '@tabler/icons-react';
import { getStoredCredentials, clearStoredCredentials } from '@/lib/auth';

export default function DashboardPage() {
  const [username, setUsername] = useState<string | null>(null);
  const [isMounted, setIsMounted] = useState(false);
  const router = useRouter();

  useEffect(() => {
    setIsMounted(true);
    const credentials = getStoredCredentials();
    if (credentials) {
      setUsername(credentials.username);
    } else {
      router.push('/login');
    }
  }, [router]);

  const handleLogout = () => {
    clearStoredCredentials();
    router.push('/login');
  };

  // Sample data for the dashboard
  const stats = [
    { title: 'Total Alpha', value: '24', change: '+12%', trend: 'up' },
    { title: 'Active Agents', value: '8', change: '+2', trend: 'up' },
    { title: 'Success Rate', value: '87%', change: '-3%', trend: 'down' },
    { title: 'Processing Time', value: '1.2s', change: '-0.3s', trend: 'up' },
  ];

  const recentActivity = [
    { id: 1, title: 'Alpha expression generated', time: '2 minutes ago', type: 'success' },
    { id: 2, title: 'Agent configuration updated', time: '15 minutes ago', type: 'info' },
    { id: 3, title: 'New data source connected', time: '1 hour ago', type: 'info' },
    { id: 4, title: 'Alpha mining completed', time: '2 hours ago', type: 'success' },
    { id: 5, title: 'Agent performance alert', time: '3 hours ago', type: 'warning' },
  ];

  if (!isMounted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black text-white flex items-center justify-center">
        <div className="animate-pulse h-8 w-8 rounded-full bg-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black text-white overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      {/* Main content */}
      <div className="relative z-10">
        {/* Header */}
        <header className="pt-8 pb-6 px-4 md:px-8">
          <div className="max-w-7xl mx-auto flex justify-between items-center">
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="backdrop-blur-md bg-white/10 p-4 rounded-xl border border-white/20"
            >
              <h1 className="text-2xl md:text-3xl font-bold">
                <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
                  DASHBOARD
                </span>
              </h1>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="flex items-center gap-4"
            >
              <div className="backdrop-blur-md bg-white/10 p-3 rounded-xl border border-white/20">
                <span className="text-blue-200">Welcome,</span>
                <span className="ml-2 font-bold">{username}</span>
              </div>
              
              <motion.button
                onClick={handleLogout}
                className="backdrop-blur-md bg-white/10 p-3 rounded-xl border border-white/20 flex items-center gap-2 hover:bg-white/20 transition-all"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <IconLogout className="h-5 w-5" />
                <span className="hidden md:inline">Logout</span>
              </motion.button>
            </motion.div>
          </div>
        </header>

        {/* Stats Section */}
        <section className="py-6 px-4 md:px-8">
          <div className="max-w-7xl mx-auto">
            <motion.h2 
              className="text-xl md:text-2xl font-bold mb-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
                PERFORMANCE METRICS
              </span>
            </motion.h2>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              {stats.map((stat, index) => (
                <motion.div
                  key={stat.title}
                  className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.1 * index }}
                >
                  <div className="flex justify-between items-start">
                    <h3 className="text-blue-200 text-sm">{stat.title}</h3>
                    <div className={`flex items-center ${stat.trend === 'up' ? 'text-green-400' : 'text-red-400'}`}>
                      {stat.trend === 'up' ? <IconArrowUpRight className="h-4 w-4" /> : <IconArrowDownRight className="h-4 w-4" />}
                      <span className="text-xs ml-1">{stat.change}</span>
                    </div>
                  </div>
                  <p className="text-3xl font-bold mt-2">{stat.value}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Recent Activity Section */}
        <section className="py-6 px-4 md:px-8">
          <div className="max-w-7xl mx-auto">
            <motion.h2 
              className="text-xl md:text-2xl font-bold mb-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
                RECENT ACTIVITY
              </span>
            </motion.h2>
            
            <motion.div
              className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <div className="space-y-4">
                {recentActivity.map((activity, index) => (
                  <motion.div
                    key={activity.id}
                    className="flex items-start gap-4 p-3 rounded-lg bg-white/5 border border-white/10"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.1 * index }}
                  >
                    <div className={`p-2 rounded-full ${
                      activity.type === 'success' ? 'bg-green-500/20 text-green-400' : 
                      activity.type === 'warning' ? 'bg-yellow-500/20 text-yellow-400' : 
                      'bg-blue-500/20 text-blue-400'
                    }`}>
                      <IconActivity className="h-5 w-5" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">{activity.title}</p>
                      <div className="flex items-center text-sm text-blue-200 mt-1">
                        <IconClock className="h-4 w-4 mr-1" />
                        <span>{activity.time}</span>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>
        </section>

        {/* Quick Actions Section */}
        <section className="py-6 px-4 md:px-8 pb-24">
          <div className="max-w-7xl mx-auto">
            <motion.h2 
              className="text-xl md:text-2xl font-bold mb-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
                QUICK ACTIONS
              </span>
            </motion.h2>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                { title: 'Create New Agent', description: 'Design and configure a new AI agent' },
                { title: 'Run Alpha Mining', description: 'Start a new alpha mining session' },
                { title: 'View Reports', description: 'Access detailed performance reports' },
                { title: 'Manage Networks', description: 'Configure agent networks and connections' },
                { title: 'Data Sources', description: 'Connect and manage data sources' },
                { title: 'Settings', description: 'Configure system and user preferences' },
              ].map((action, index) => (
                <motion.div
                  key={action.title}
                  className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20 cursor-pointer hover:bg-white/20 transition-all"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.1 * index }}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <h3 className="text-lg font-bold mb-2">{action.title}</h3>
                  <p className="text-blue-200 text-sm">{action.description}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>
      </div>

      {/* Floating Navigation Dock */}
      {isMounted && (
        <FloatingDock items={sharedNavItems} />
      )}
    </div>
  );
} 