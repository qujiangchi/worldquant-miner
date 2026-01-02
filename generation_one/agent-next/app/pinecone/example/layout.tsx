import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Pinecone Integration Example | WorldQuant Miner',
  description: 'Example of integrating Pinecone vector database with WorldQuant Miner',
  keywords: ['Pinecone', 'vector database', 'integration', 'example', 'WorldQuant', 'Miner'],
  openGraph: {
    title: 'Pinecone Integration Example | WorldQuant Miner',
    description: 'Example of integrating Pinecone vector database with WorldQuant Miner',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Pinecone Integration Example | WorldQuant Miner',
    description: 'Example of integrating Pinecone vector database with WorldQuant Miner',
  },
};

export default function PineconeExampleLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted">
      {children}
    </div>
  );
} 