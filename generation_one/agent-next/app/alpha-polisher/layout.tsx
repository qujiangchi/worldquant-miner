import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Alpha Polisher | WorldQuant Alpha Generator',
  description: 'Polish your alpha expressions or generate new ideas based on your suggestions using AI-powered tools.',
  keywords: 'alpha polisher, worldquant, alpha generation, quantitative finance, alpha ideas',
  openGraph: {
    title: 'Alpha Polisher | WorldQuant Alpha Generator',
    description: 'Polish your alpha expressions or generate new ideas based on your suggestions using AI-powered tools.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Alpha Polisher | WorldQuant Alpha Generator',
    description: 'Polish your alpha expressions or generate new ideas based on your suggestions using AI-powered tools.',
  },
};

export default function AlphaPolisherLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-background">
      {children}
    </div>
  );
} 