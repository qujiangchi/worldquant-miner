import { NextRequest, NextResponse } from 'next/server';

// Base URL for WorldQuant Brain API
const API_BASE_URL = 'https://api.worldquantbrain.com';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { username, password } = body;
    
    if (!username || !password) {
      return NextResponse.json(
        { error: 'Username and password are required' },
        { status: 400 }
      );
    }
    
    // Create base64 encoded auth string
    const authString = `${username}:${password}`;
    const base64Auth = Buffer.from(authString).toString('base64');
    
    // Make authentication request to WorldQuant API
    const response = await fetch(`${API_BASE_URL}/authentication`, {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${base64Auth}`,
        'Content-Type': 'application/json',
      },
    });
    
    console.log(response)

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        { error: `Authentication failed: ${response.status} ${errorText}` },
        { status: response.status }
      );
    }
    
    // Parse the response to get the token
    const data = await response.json();


    const token = data.token || data.access_token;
    
    if (!token) {
      return NextResponse.json(
        { error: 'No token received from WorldQuant API' },
        { status: 500 }
      );
    }
    
    // Return the token and username
    return NextResponse.json({
      token,
      username
    });
  } catch (error) {
    console.error('Authentication error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
} 