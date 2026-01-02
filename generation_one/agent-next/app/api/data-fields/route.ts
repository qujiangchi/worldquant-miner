import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    // Extract the JWT token from the request body
    const { jwtToken, dataset, limit, instrumentType, region, universe, delay, offset } = await request.json();
    
    if (!jwtToken) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }
    
    // Construct the URL with query parameters
    const url = new URL('https://api.worldquantbrain.com/data-fields');
    
    // Add query parameters
    if (dataset) url.searchParams.append('dataset.id', dataset);
    if (limit) url.searchParams.append('limit', limit);
    if (instrumentType) url.searchParams.append('instrumentType', instrumentType);
    if (region) url.searchParams.append('region', region);
    if (universe) url.searchParams.append('universe', universe);
    if (delay) url.searchParams.append('delay', delay);
    // offset
    if (offset) url.searchParams.append('offset', offset);
    
    // Make the request to the WorldQuant API
    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: {
        'Cookie': `t=${jwtToken}`,
      },
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('WorldQuant API error:', errorText);
      return NextResponse.json({ error: 'Failed to fetch data fields' }, { status: response.status });
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
    console.error('Error in data-fields API route:', error);
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