# Object Detection PoC - Real-time Spatial Awareness

A mobile proof-of-concept application that performs real-time object detection via the phone's camera, estimates distance and direction, and provides spatial alerts using only open-source tools.

## Features

🎯 **Real-time Object Detection** - Live camera feed with TensorFlow.js COCO-SSD model  
📏 **Distance Estimation** - Calculates approximate distance based on object size  
🧭 **Direction Detection** - Determines left/center/right positioning  
🚨 **Spatial Alerts** - Real-time notifications like "person approaching from right ~25m"  
🔊 **Voice Alerts** - Configurable audio announcements for detected objects  
⚙️ **Settings Panel** - Comprehensive configuration interface  
📱 **Mobile Optimized** - Responsive design with Capacitor mobile integration  
⚡ **High Performance** - Optimized for mobile devices with FPS monitoring  

## Technology Stack

- **Frontend**: React + TypeScript + Vite
- **UI**: Tailwind CSS + shadcn/ui components
- **Object Detection**: TensorFlow.js + COCO-SSD model
- **Mobile**: Capacitor for native capabilities
- **Design**: Dark theme optimized for camera interfaces

## Getting Started

### Web Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### Mobile Development

To run on physical device or emulator:

1. **Export to GitHub** via the "Export to Github" button
2. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd <project-name>
   npm install
   ```

3. **Add mobile platforms**:
   ```bash
   # For iOS (requires macOS + Xcode)
   npx cap add ios
   npx cap update ios
   
   # For Android (requires Android Studio)
   npx cap add android  
   npx cap update android
   ```

4. **Build and sync**:
   ```bash
   npm run build
   npx cap sync
   ```

5. **Run on device**:
   ```bash
   # iOS (requires macOS + Xcode)
   npx cap run ios
   
   # Android (requires Android Studio)
   npx cap run android
   ```

## How It Works

### Object Detection
- Uses TensorFlow.js with COCO-SSD model for real-time detection
- Detects 80 different object classes (person, car, bicycle, etc.)
- Runs entirely in the browser/webview

### Distance Estimation
- Calculates distance using object pixel height vs known real-world dimensions
- Formula: `Distance = (Real Height × Focal Length) / Pixel Height`
- Provides estimates between 1-200 meters

### Spatial Awareness
- Determines object direction based on bounding box position
- Generates contextual alerts: "person very close on the left (~8m)"
- Alert severity based on proximity (danger < 10m, warning < 25m)

### Performance
- Real-time FPS monitoring
- Optimized canvas rendering
- Mobile-first responsive design

### Voice Alerts
- Configurable audio notifications for detected objects
- Multiple alert types: immediate, summary, or both
- Customizable voice settings: volume, rate, pitch
- Smart announcements with distance and direction information
- Accessibility-focused for visually impaired users

## Camera Permissions

The app requires camera access. On mobile devices:
- **iOS**: Permissions handled automatically via Capacitor
- **Android**: Permissions handled automatically via Capacitor
- **Web**: Browser will prompt for camera access

## Model Details

- **Model**: COCO-SSD (Common Objects in Context - Single Shot Detection)
- **Classes**: 80 different object types
- **Accuracy**: Optimized for mobile performance vs accuracy balance
- **Size**: ~27MB download (cached after first load)

## Troubleshooting

### Camera Not Working
- Check browser/app permissions
- Ensure HTTPS (required for camera access)
- Try different browsers if on web

### Low Performance
- Reduce video resolution in camera constraints
- Ensure good lighting conditions
- Close other apps on mobile device

### Model Loading Issues
- Check internet connection (required for first load)
- Clear browser cache if issues persist
- Check console for TensorFlow.js errors

## Development Notes

- Hot reload enabled for mobile development via Capacitor server config
- All colors use semantic tokens from design system
- Responsive design works across all screen sizes
- Built-in error handling and user feedback

## Future Enhancements

- Custom model training with TensorFlow Lite Model Maker
- Depth estimation using dual cameras
- Audio alerts for accessibility
- Recording and playback of detection sessions
- Integration with device sensors (accelerometer, compass)

---

Built with ❤️ using Lovable and open-source technologies.