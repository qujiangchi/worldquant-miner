import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Pinecone Vector Database | WorldQuant Miner',
  description: 'Interact with your Pinecone vector database to store, query, and manage vector embeddings.',
  keywords: 'pinecone, vector database, embeddings, semantic search, worldquant, alpha, finance',
  openGraph: {
    title: 'Pinecone Vector Database | WorldQuant Miner',
    description: 'Interact with your Pinecone vector database to store, query, and manage vector embeddings.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Pinecone Vector Database | WorldQuant Miner',
    description: 'Interact with your Pinecone vector database to store, query, and manage vector embeddings.',
  },
};

export default function PineconeLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white">
      {children}
    </div>
  );
} 