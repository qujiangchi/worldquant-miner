'use client';

import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import dynamic from 'next/dynamic';
import { FloatingDock } from '@/components/ui/floating-dock';
import { sharedNavItems } from '@/components/ui/shared-navigation';
import { 
  IconHome, 
  IconChartBar, 
  IconBrain, 
  IconSettings, 
  IconUser,
  IconArrowRight,
  IconSpider
} from '@tabler/icons-react';
import Link from 'next/link';

// Dynamically import the WorldMap component with no SSR to avoid hydration issues
const WorldMap = dynamic(() => import('@/components/ui/world-map').then(mod => mod.WorldMap), {
  ssr: false,
  loading: () => (
    <div className="w-full aspect-[2/1] dark:bg-black bg-white rounded-lg relative font-sans">
      <div className="h-full w-full flex items-center justify-center">
        <div className="animate-pulse h-8 w-8 rounded-full bg-gray-300 dark:bg-gray-700"></div>
      </div>
    </div>
  )
});

export default function HomePage() {
  const [isHovered, setIsHovered] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  // Set mounted state after hydration
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Sample data for the world map
  const mapDots = [
    { start: { lat: 40.7128, lng: -74.0060 }, end: { lat: 51.5074, lng: -0.1278 } },
    { start: { lat: 35.6762, lng: 139.6503 }, end: { lat: 22.3193, lng: 114.1694 } },
    { start: { lat: 48.8566, lng: 2.3522 }, end: { lat: 55.7558, lng: 37.6173 } },
    { start: { lat: -33.8688, lng: 151.2093 }, end: { lat: 1.3521, lng: 103.8198 } },
  ];

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
        {/* Hero section */}
        <section className="pt-20 pb-16 px-4 md:px-8">
          <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
              <div className="space-y-8">
                <motion.h1 
                  className="text-5xl md:text-7xl font-bold tracking-tight"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
                    INTERACTIVE AGENT
                  </span>
                  <br />
                  <span className="text-white">
                    WORKBENCH
                  </span>
                </motion.h1>
                
                <motion.p 
                  className="text-xl md:text-2xl text-gray-300 max-w-lg"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  Mine alpha for WorldQuant with our powerful interactive agent workbench.
                </motion.p>
                
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.4 }}
                >
                  <Link href="/web-miner">
                    <motion.button
                      className="px-8 py-4 bg-white text-black font-bold rounded-lg flex items-center gap-2 group"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onHoverStart={() => setIsHovered(true)}
                      onHoverEnd={() => setIsHovered(false)}
                    >
                      Try Web Crawler
                      <IconArrowRight 
                        className="h-5 w-5 transition-transform duration-300 group-hover:translate-x-1" 
                      />
                    </motion.button>
                  </Link>
                </motion.div>
              </div>
              
              {isMounted && (
                <motion.div
                  className="relative"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.7, delay: 0.3 }}
                >
                  <div className="backdrop-blur-md bg-white/10 p-4 rounded-2xl border border-white/20 shadow-2xl">
                    <WorldMap dots={mapDots} lineColor="#3b82f6" />
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        </section>

        {/* Features section */}
        <section className="py-16 px-4 md:px-8">
          <div className="max-w-7xl mx-auto">
            <motion.h2 
              className="text-3xl md:text-4xl font-bold text-center mb-12"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
                POWERFUL FEATURES
              </span>
            </motion.h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                {
                  title: "Web Crawler",
                  description: "Extract insights from research papers and generate alpha ideas using WorldQuant data."
                },
                {
                  title: "Advanced Analytics",
                  description: "Leverage powerful analytics tools to identify market opportunities and trends."
                },
                {
                  title: "Real-time Data",
                  description: "Access real-time market data and insights to make informed decisions."
                }
              ].map((feature, index) => (
                <motion.div
                  key={feature.title}
                  className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20"
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                  <p className="text-gray-300">{feature.description}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>
      </div>

    </div>
  );
}

