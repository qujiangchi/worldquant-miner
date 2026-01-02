'use client';
/**
 * Note: Use position fixed according to your needs
 * Desktop navbar is better positioned at the bottom
 * Mobile navbar is better positioned at bottom right.
 **/
 
import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

interface DockItem {
  title: string;
  icon: React.ReactNode;
  href: string;
}

interface FloatingDockProps {
  items: DockItem[];
}

export function FloatingDock({ items }: FloatingDockProps) {
  const pathname = usePathname();

  return (
    <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50">
      <div className="backdrop-blur-md bg-white/10 p-2 rounded-full border border-white/20 flex items-center space-x-2">
        {items.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`p-3 rounded-full transition-colors ${
                isActive
                  ? 'bg-blue-500 text-white'
                  : 'text-white/70 hover:bg-white/10 hover:text-white'
              }`}
              title={item.title}
            >
              {item.icon}
            </Link>
          );
        })}
      </div>
    </div>
  );
} 