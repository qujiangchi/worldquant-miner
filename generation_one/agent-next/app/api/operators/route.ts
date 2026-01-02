import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    // Extract the JWT token from the request body
    const { jwtToken } = await request.json();
    
    if (!jwtToken) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }
    
    // Make the request to the WorldQuant API
    const response = await fetch('https://api.worldquantbrain.com/operators', {
      method: 'GET',
      headers: {
        'Cookie': `t=${jwtToken}`,
      },
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('WorldQuant API error:', errorText);
      return NextResponse.json({ error: 'Failed to fetch operators' }, { status: response.status });
    }
    
    const data = await response.json();
    
    // Add CORS headers to the response
    const headers = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };
    
    return NextResponse.json(data, { headers });
  } catch (error) {
    console.error('Error in operators API route:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

// Handle OPTIONS requests for CORS preflight
export async function OPTIONS() {
  return NextResponse.json({}, {
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
} 