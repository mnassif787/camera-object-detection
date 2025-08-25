import React from 'react';

const TestPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-background p-8">
      <h1 className="text-2xl font-bold mb-4">Test Page</h1>
      <p className="text-muted-foreground">
        This is a simple test page to verify the application is working.
      </p>
      <div className="mt-4 p-4 bg-card rounded-lg">
        <h2 className="text-lg font-semibold mb-2">Status</h2>
        <ul className="space-y-1 text-sm">
          <li>✅ React is working</li>
          <li>✅ TypeScript is working</li>
          <li>✅ Tailwind CSS is working</li>
          <li>✅ shadcn/ui components are working</li>
        </ul>
      </div>
    </div>
  );
};

export default TestPage;