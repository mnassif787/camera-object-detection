import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'app.lovable.db6c5374057a4b0ea4003fe639402b34',
  appName: 'Object Detection PoC',
  webDir: 'dist',
  server: {
    url: 'https://db6c5374-057a-4b0e-a400-3fe639402b34.lovableproject.com?forceHideBadge=true',
    cleartext: true
  },
  plugins: {
    Camera: {
      permissions: ["camera"]
    }
  }
};

export default config;