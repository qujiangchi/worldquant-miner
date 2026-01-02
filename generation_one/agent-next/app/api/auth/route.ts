import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { username, password, jwtToken } = await request.json();

    // Make request to WorldQuant API
    const response = await fetch('https://api.worldquantbrain.com/operators', {
      method: 'GET',
      headers: {
        'Authorization': `Basic ${btoa(`${username}:${password}`)}`,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Host': 'api.worldquantbrain.com',
        'Connection': 'keep-alive',
        'Origin': 'https://platform.worldquantbrain.com',
        'Referer': 'https://platform.worldquantbrain.com/',
        ...(jwtToken ? { 'Cookie': `t=${jwtToken}` } : {}),
      },
      credentials: 'include',
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: 'Authentication failed' },
        { status: 401 }
      );
    }

    // Get the set-cookie header from the response
    const setCookieHeader = response.headers.get('set-cookie');
    
    console.log(response)
    console.log(setCookieHeader)
    // Create a new response with the data and the set-cookie header
    const newResponse = NextResponse.json(
      { 
        success: true,
        jwtToken: jwtToken || (setCookieHeader ? setCookieHeader.split(';')[0].split('=')[1] : null)
      },
      { status: 200 }
    );
    
    // If we have a set-cookie header, add it to our response
    if (setCookieHeader) {
      newResponse.headers.set('Set-Cookie', setCookieHeader);
    }
    
    return newResponse;
  } catch (error) {
    console.error('Authentication error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// Handle OPTIONS requests for CORS preflight
export async function OPTIONS(request: Request) {
  const response = new NextResponse(null, { status: 204 });
  
  // Add CORS headers
  response.headers.set('Access-Control-Allow-Origin', '*');
  response.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, Cookie');
  
  return response;
} 