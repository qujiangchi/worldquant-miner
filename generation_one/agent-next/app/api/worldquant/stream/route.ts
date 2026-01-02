import { NextRequest } from 'next/server';

// Base URL for WorldQuant Brain API
const API_BASE_URL = 'https://api.worldquantbrain.com';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const endpoint = searchParams.get('endpoint');
  
  if (!endpoint) {
    return new Response('Endpoint parameter is required', { status: 400 });
  }
  
  // Extract other query parameters
  const params: Record<string, any> = {};
  searchParams.forEach((value, key) => {
    if (key !== 'endpoint') {
      params[key] = value;
    }
  });
  
  // Get credentials from Authorization header or cookies
  const authHeader = request.headers.get('Authorization');
  
  if (!authHeader || !authHeader.startsWith('Basic ')) {
    return new Response('data: {"error":"Authentication required"}\n\n', {
      status: 401,
      headers: {
        'Content-Type': 'text/event-stream',
      },
    });
  }
  
  // Create a new ReadableStream for SSE
  const stream = new ReadableStream({
    async start(controller) {
      try {
        // Use the Authorization header directly
        const base64Auth = authHeader.replace('Basic ', '');
        
        const url = new URL(`${API_BASE_URL}${endpoint}`);
        
        // Add query parameters
        Object.entries(params).forEach(([key, value]) => {
          if (value !== undefined && value !== null) {
            url.searchParams.append(key, String(value));
          }
        });
        
        // Send initial message
        controller.enqueue(new TextEncoder().encode(`data: {"status":"connecting","message":"Connecting to WorldQuant API..."}\n\n`));
        
        // Make the API request
        const response = await fetch(url.toString(), {
          headers: {
            'Authorization': `Basic ${base64Auth}`,
            'Content-Type': 'application/json',
          },
        });
        
        if (!response.ok) {
          const errorText = await response.text();
          controller.enqueue(new TextEncoder().encode(`data: {"error":"WorldQuant API error: ${response.status} ${errorText}"}\n\n`));
          controller.close();
          return;
        }
        
        // Send success message
        controller.enqueue(new TextEncoder().encode(`data: {"status":"connected","message":"Connected to WorldQuant API"}\n\n`));
        
        // Parse the response
        const data = await response.json();
        
        // Send the data
        controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(data)}\n\n`));
        
        // Close the stream
        controller.close();
      } catch (error) {
        console.error('Error in WorldQuant API stream:', error);
        controller.enqueue(new TextEncoder().encode(`data: {"error":"${error instanceof Error ? error.message : 'Unknown error'}"}\n\n`));
        controller.close();
      }
    },
  });
  
  // Return the stream as a Server-Sent Events response
  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
} 