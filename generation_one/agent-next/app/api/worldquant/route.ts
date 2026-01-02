import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';

// Base URL for WorldQuant Brain API
const API_BASE_URL = 'https://api.worldquantbrain.com';

// GET handler for data fields
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const endpoint = searchParams.get('endpoint');
    
    if (!endpoint) {
      return NextResponse.json({ error: 'Endpoint parameter is required' }, { status: 400 });
    }
    
    // Get credentials from the request headers
    const authHeader = request.headers.get('Authorization');
    
    if (!authHeader || !authHeader.startsWith('Basic ')) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }
    
    // Forward the request to the WorldQuant API
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: 'API request failed' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('API request error:', error);
    return NextResponse.json(
      { error: 'API request failed' },
      { status: 500 }
    );
  }
}

// POST handler for generating alpha ideas
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

    // Create Basic Auth header
    const authHeader = 'Basic ' + Buffer.from(username + ':' + password).toString('base64');

    // Test the credentials by making a request to the API
    const response = await fetch(`${API_BASE_URL}/operators`, {
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: 'Authentication failed' },
        { status: 401 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Authentication error:', error);
    return NextResponse.json(
      { error: 'Authentication failed' },
      { status: 500 }
    );
  }
}

// Helper function to generate alpha ideas (simulated)
function generateAlphaIdeas(fields: string[], operators: string[] = []) {
  // This is a placeholder - in a real implementation, this would use the WorldQuant API
  // to generate alpha ideas based on the selected fields and operators
  
  const ideas = [];
  const fieldNames = fields.map(field => field.split('_').pop() || field);
  
  // Generate a few sample ideas
  for (let i = 0; i < 3; i++) {
    const randomFields = [...fields].sort(() => Math.random() - 0.5).slice(0, 2);
    const fieldName1 = randomFields[0].split('_').pop() || randomFields[0];
    const fieldName2 = randomFields[1]?.split('_').pop() || randomFields[1];
    
    const idea = {
      id: `alpha${i + 1}`,
      name: `${fieldName1} ${fieldName2 ? `& ${fieldName2}` : ''} Strategy`,
      description: `Alpha strategy based on ${fieldName1}${fieldName2 ? ` and ${fieldName2}` : ''}`,
      formula: `rank(${fieldName1}) * ${Math.random() > 0.5 ? '-1 * ' : ''}rank(${fieldName2 || fieldName1})`,
      dataFields: randomFields,
      performance: {
        sharpe: (1 + Math.random() * 2).toFixed(2),
        returns: (10 + Math.random() * 15).toFixed(1),
        drawdown: (-5 - Math.random() * 10).toFixed(1)
      }
    };
    
    ideas.push(idea);
  }
  
  return ideas;
} 