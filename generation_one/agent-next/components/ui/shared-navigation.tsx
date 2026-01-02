import { 
  IconHome, 
  IconChartBar, 
  IconBrain, 
  IconSettings, 
  IconUser,
  IconWorld,
  IconDatabase,
  IconChartDots,
  IconNetwork,
  IconGitBranch,
  IconActivity,
  IconRobot
} from '@tabler/icons-react';

export const sharedNavItems = [
  { title: 'Home', icon: <IconHome className="h-5 w-5" />, href: '/' },
  { title: 'Dashboard', icon: <IconChartBar className="h-5 w-5" />, href: '/dashboard' },
  { title: 'Web Miner', icon: <IconWorld className="h-5 w-5" />, href: '/web-miner' },
  { title: 'Alpha Polisher', icon: <IconBrain className="h-5 w-5" />, href: '/alpha-polisher' },
  { title: 'Results', icon: <IconChartDots className="h-5 w-5" />, href: '/results' },
  { title: 'Simulation', icon: <IconActivity className="h-5 w-5" />, href: '/simulation' },
  { title: 'Pinecone', icon: <IconDatabase className="h-5 w-5" />, href: '/pinecone' },
  { title: 'Networks', icon: <IconNetwork className="h-5 w-5" />, href: '/networks' },
  { title: 'Network Designer', icon: <IconGitBranch className="h-5 w-5" />, href: '/network-designer' },
  { title: 'Brain', icon: <IconBrain className="h-5 w-5" />, href: '/brain' },
  { title: 'MCP Agent', icon: <IconRobot className="h-5 w-5" />, href: '/mcp-agent' },
  { title: 'Settings', icon: <IconSettings className="h-5 w-5" />, href: '/settings' },
  { title: 'Profile', icon: <IconUser className="h-5 w-5" />, href: '/profile' },
  { title: 'A2A', icon: <IconBrain className="h-5 w-5" />, href: '/a2a' },
]; 