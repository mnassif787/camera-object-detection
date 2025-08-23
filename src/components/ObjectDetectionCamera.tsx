import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Button } from '@/components/ui/button';
import { Camera, Square, Volume2, VolumeX } from 'lucide-react';

interface Detection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
  distance: number;
}

const ObjectDetectionCamera: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const animationRef = useRef<number>();

  const [isLoading, setIsLoading] = useState(true);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);

  // Initialize TensorFlow.js and load model
  useEffect(() => {
    const initializeTensorFlow = async () => {
      try {
        console.log('Starting TensorFlow.js initialization...');
        await tf.ready();
        console.log('TensorFlow.js initialized successfully');
        
        console.log('Loading COCO-SSD model...');
        const model = await cocoSsd.load();
        console.log('COCO-SSD model loaded successfully:', model);
        modelRef.current = model;
        setModelLoaded(true);
        setIsLoading(false);
        
        console.log('Model loaded and ready');
      } catch (error) {
        console.error('Error loading model:', error);
        setIsLoading(false);
      }
    };

    initializeTensorFlow();
  }, []);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      console.log('Starting camera...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          frameRate: { ideal: 30, max: 30 }
        }
      });
      console.log('Camera stream obtained:', stream);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        
        videoRef.current.onloadedmetadata = () => {
          console.log('Video metadata loaded, starting detection...');
          if (videoRef.current) {
            videoRef.current.play();
            setIsDetecting(true);
            startDetection();
          }
        };
      }
    } catch (error) {
      console.error('Camera access error:', error);
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    setIsDetecting(false);
    setDetections([]);
  }, []);

  // Distance estimation using focal length approximation
  const estimateDistance = (bbox: [number, number, number, number], className: string): number => {
    const [x, y, width, height] = bbox;
    
    // Known real-world dimensions for common objects (in meters)
    const objectDimensions: { [key: string]: number } = {
      'person': 0.5,      // Average person width
      'car': 1.8,         // Average car width
      'bicycle': 0.6,     // Bicycle width
      'motorcycle': 0.8,  // Motorcycle width
      'bus': 2.5,         // Bus width
      'truck': 2.5,       // Truck width
      'dog': 0.3,         // Dog width
      'cat': 0.2,         // Cat width
      'chair': 0.5,       // Chair width
      'table': 1.2,       // Table width
      'bottle': 0.08,     // Bottle width
      'cup': 0.08,        // Cup width
      'laptop': 0.35,     // Laptop width
      'tv': 1.0,          // TV width
      'cell phone': 0.07, // Phone width
      'book': 0.15,       // Book width
      'bowl': 0.15,       // Bowl width
      'apple': 0.08,      // Apple width
      'banana': 0.03,     // Banana width
      'airplane': 35.0,   // Airplane width
      'train': 3.0,       // Train width
      'boat': 2.0,        // Boat width
      'fire hydrant': 0.3, // Fire hydrant width
      'stop sign': 0.3,   // Stop sign width
      'bench': 1.5,       // Bench width
      'bird': 0.15,       // Bird width
      'horse': 1.0,       // Horse width
      'sheep': 0.8,       // Sheep width
      'cow': 1.2,         // Cow width
      'elephant': 2.5,    // Elephant width
      'bear': 1.5,        // Bear width
      'zebra': 1.0,       // Zebra width
      'giraffe': 0.8,     // Giraffe width
      'backpack': 0.3,    // Backpack width
      'umbrella': 0.8,    // Umbrella width
      'handbag': 0.3,     // Handbag width
      'suitcase': 0.5,    // Suitcase width
      'frisbee': 0.25,    // Frisbee width
      'skis': 0.1,        // Skis width
      'snowboard': 0.3,   // Snowboard width
      'sports ball': 0.22, // Sports ball width
      'kite': 0.8,        // Kite width
      'baseball bat': 0.05, // Baseball bat width
      'baseball glove': 0.25, // Baseball glove width
      'skateboard': 0.8,  // Skateboard width
      'surfboard': 2.0,   // Surfboard width
      'tennis racket': 0.3, // Tennis racket width
      'wine glass': 0.08, // Wine glass width
      'fork': 0.02,       // Fork width
      'knife': 0.02,      // Knife width
      'spoon': 0.02,      // Spoon width
      'sandwich': 0.15,   // Sandwich width
      'orange': 0.08,     // Orange width
      'broccoli': 0.15,   // Broccoli width
      'carrot': 0.03,     // Carrot width
      'hot dog': 0.15,    // Hot dog width
      'pizza': 0.3,       // Pizza width
      'donut': 0.1,       // Donut width
      'cake': 0.25,       // Cake width
      'bed': 1.4,         // Bed width
      'dining table': 1.8, // Dining table width
      'toilet': 0.4,      // Toilet width
      'remote': 0.15,     // Remote width
      'microwave': 0.5,   // Microwave width
      'oven': 0.6,        // Oven width
      'toaster': 0.3,     // Toaster width
      'sink': 0.5,        // Sink width
      'refrigerator': 0.8, // Refrigerator width
      'clock': 0.3,        // Clock width
      'vase': 0.15,       // Vase width
      'scissors': 0.15,   // Scissors width
      'teddy bear': 0.3,  // Teddy bear width
      'hair drier': 0.2,  // Hair drier width
      'toothbrush': 0.02, // Toothbrush width
    };

    // Get known width for this object type, default to 0.5m if unknown
    const knownWidth = objectDimensions[className] || 0.5;
    
    // Focal length approximation (adjust experimentally for your camera)
    const focalLength = 700;
    
    // Calculate distance using focal length formula: distance = (known_width * focal_length) / perceived_width
    const distance = (knownWidth * focalLength) / width;
    
    // Clamp distance to reasonable bounds (0.5m to 50m)
    const clampedDistance = Math.max(0.5, Math.min(50, distance));
    
    return Math.round(clampedDistance * 10) / 10; // Round to 1 decimal place
  };

  // Get color based on distance (exactly as specified)
  const getDistanceColor = (distance: number): string => {
    if (distance < 3) return '#FF0000';      // Red if distance < 3m
    if (distance <= 5) return '#FFFF00';     // Yellow if distance between 3m and 5m
    return '#00FF00';                        // Green if distance > 5m
  };

  // Simple speech for detected objects
  const speakDetection = useCallback((detection: Detection) => {
    if (!speechEnabled) return;
    
    const message = `${detection.class} detected at ${detection.distance} meters`;
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
    
    console.log('Voice alert:', message);
  }, [speechEnabled]);

  // Main detection loop - optimized for 30+ FPS
  const startDetection = useCallback(() => {
    console.log('Starting detection loop...');
    
    let lastTime = Date.now();
    let frameCount = 0;
    let lastDetectionTime = 0;
    const detectionInterval = 100; // Run detection every 100ms for smooth 30+ FPS

    const detect = async () => {
      if (!videoRef.current || !canvasRef.current || !modelRef.current) {
        console.log('Missing refs, stopping detection');
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const currentTime = Date.now();

      if (!ctx) {
        console.log('No canvas context');
        animationRef.current = requestAnimationFrame(detect);
        return;
      }

      // Ensure video has dimensions
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        console.log('Waiting for video dimensions...');
        animationRef.current = requestAnimationFrame(detect);
        return;
      }

      // Set canvas size to match video dimensions EXACTLY
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        console.log('Canvas resized to match video:', canvas.width, 'x', canvas.height);
      }

      // Update canvas CSS dimensions to match video display size
      const videoRect = video.getBoundingClientRect();
      canvas.style.width = videoRect.width + 'px';
      canvas.style.height = videoRect.height + 'px';
      canvas.style.top = '0px';
      canvas.style.left = '0px';

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw a comprehensive test pattern to verify canvas is working
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 4;
      ctx.strokeRect(10, 10, 150, 80);
      ctx.fillStyle = '#00FF00';
      ctx.font = 'bold 18px Arial';
      ctx.fillText('CANVAS OK', 20, 50);
      ctx.fillText(`Size: ${canvas.width}x${canvas.height}`, 20, 75);
      
      // Draw grid lines for better object positioning reference
      ctx.strokeStyle = '#333333';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      
      // Vertical grid lines
      for (let i = 0; i <= canvas.width; i += 100) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, canvas.height);
        ctx.stroke();
      }
      
      // Horizontal grid lines
      for (let i = 0; i <= canvas.height; i += 100) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(canvas.width, i);
        ctx.stroke();
      }
      
      ctx.setLineDash([]); // Reset line dash
      
      // Draw center cross for reference
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      ctx.strokeStyle = '#FF0000';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(centerX - 20, centerY);
      ctx.lineTo(centerX + 20, centerY);
      ctx.moveTo(centerX, centerY - 20);
      ctx.lineTo(centerX, centerY + 20);
      ctx.stroke();
      
      // Draw center dot
      ctx.fillStyle = '#FF0000';
      ctx.beginPath();
      ctx.arc(centerX, centerY, 5, 0, 2 * Math.PI);
      ctx.fill();

      try {
        // Run object detection at controlled intervals
        if (currentTime - lastDetectionTime >= detectionInterval) {
          console.log('Running object detection...');
          
          const predictions = await modelRef.current.detect(video);
          const filteredPredictions = predictions.filter(prediction => prediction.score > 0.3);
          
          if (filteredPredictions.length > 0) {
            console.log('Objects detected:', filteredPredictions.map(p => `${p.class}:${Math.round(p.score*100)}%`));
          }
          
          // Process detections with distance estimation
          const newDetections = filteredPredictions.map(prediction => {
            const distance = estimateDistance(prediction.bbox, prediction.class);
            return {
              bbox: prediction.bbox,
              class: prediction.class,
              score: prediction.score,
              distance
            };
          });
          
          setDetections(newDetections);
          
          // Speak about new detections
          newDetections.forEach(detection => {
            speakDetection(detection);
          });
          
          lastDetectionTime = currentTime;
        }

        // Draw bounding boxes and labels for all detected objects
        detections.forEach((detection, index) => {
          const [x, y, width, height] = detection.bbox;
          const color = getDistanceColor(detection.distance);
          
          console.log(`Drawing object ${index}:`, detection.class, 'at', x, y, width, height, 'distance:', detection.distance);
          
          // Validate bbox coordinates
          if (x < 0 || y < 0 || x + width > canvas.width || y + height > canvas.height) {
            console.warn('Bbox coordinates out of bounds:', { x, y, width, height, canvasWidth: canvas.width, canvasHeight: canvas.height });
            return;
          }
          
          // Draw filled colored rectangle OVER the object (semi-transparent overlay)
          ctx.fillStyle = color + '40'; // Add 40 for 25% opacity
          ctx.fillRect(x, y, width, height);
          
          // Draw main bounding box with distance-based color - make it VERY visible
          ctx.strokeStyle = color;
          ctx.lineWidth = 8; // Much thicker lines for better visibility
          ctx.strokeRect(x, y, width, height);
          
          // Add a bright white border around the bbox for extra visibility
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 3;
          ctx.strokeRect(x-3, y-3, width+6, height+6);
          
          // Draw inner highlight rectangle for better object definition
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.strokeRect(x+4, y+4, width-8, height-8);
          
          // Draw label with object name + distance above the bounding box
          const label = `${detection.class} - ${detection.distance}m`;
          ctx.fillStyle = color;
          ctx.font = 'bold 20px Arial'; // Larger font for better visibility
          
          // Position label above the object, or below if too close to top
          const labelY = y > 30 ? y - 20 : y + height + 30;
          
          // Draw label background for better readability
          const labelMetrics = ctx.measureText(label);
          const labelWidth = labelMetrics.width + 20;
          const labelHeight = 30;
          const labelX = x;
          const labelBgY = labelY - 20;
          
          // Label background with color matching the object
          ctx.fillStyle = color + 'CC'; // 80% opacity
          ctx.fillRect(labelX, labelBgY, labelWidth, labelHeight);
          
          // Label border
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 2;
          ctx.strokeRect(labelX, labelBgY, labelWidth, labelHeight);
          
          // Label text
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(label, x + 10, labelY);
          
          // Add a large colored dot to indicate distance level (top-right corner)
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x + width - 20, y + 20, 12, 0, 2 * Math.PI);
          ctx.fill();
          
          // Add white border to the distance dot for better visibility
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 3;
          ctx.stroke();
          
          // Add distance text inside the dot
          ctx.fillStyle = '#FFFFFF';
          ctx.font = 'bold 12px Arial';
          ctx.textAlign = 'center';
          ctx.fillText(Math.round(detection.distance), x + width - 20, y + 25);
          ctx.textAlign = 'left'; // Reset text alignment
          
          // Draw a center marker cross for better object identification
          const centerX = x + width/2;
          const centerY = y + height/2;
          const crossSize = 15;
          
          // Center cross with white background
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.moveTo(centerX - crossSize, centerY);
          ctx.lineTo(centerX + crossSize, centerY);
          ctx.moveTo(centerX, centerY - crossSize);
          ctx.lineTo(centerX, centerY + crossSize);
          ctx.stroke();
          
          // Center cross with colored overlay
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(centerX - crossSize, centerY);
          ctx.lineTo(centerX + crossSize, centerY);
          ctx.moveTo(centerX, centerY - crossSize);
          ctx.lineTo(centerX, centerY + crossSize);
          ctx.stroke();
          
          // Add corner indicators for better object definition
          const cornerSize = 8;
          const cornerColor = '#FFFFFF';
          
          // Top-left corner
          ctx.strokeStyle = cornerColor;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(x, y + cornerSize);
          ctx.lineTo(x, y);
          ctx.lineTo(x + cornerSize, y);
          ctx.stroke();
          
          // Top-right corner
          ctx.beginPath();
          ctx.moveTo(x + width - cornerSize, y);
          ctx.lineTo(x + width, y);
          ctx.lineTo(x + width, y + cornerSize);
          ctx.stroke();
          
          // Bottom-left corner
          ctx.beginPath();
          ctx.moveTo(x, y + height - cornerSize);
          ctx.lineTo(x, y + height);
          ctx.lineTo(x + cornerSize, y + height);
          ctx.stroke();
          
          // Bottom-right corner
          ctx.beginPath();
          ctx.moveTo(x + width - cornerSize, y + height);
          ctx.lineTo(x + width, y + height);
          ctx.lineTo(x + width, y + height - cornerSize);
          ctx.stroke();
          
          // Add object type indicator (small colored square)
          ctx.fillStyle = color;
          ctx.fillRect(x + 5, y + 5, 15, 15);
          
          // Add white border to object type indicator
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 2;
          ctx.strokeRect(x + 5, y + 5, 15, 15);
          
          // Add first letter of object type
          ctx.fillStyle = '#FFFFFF';
          ctx.font = 'bold 12px Arial';
          ctx.fillText(detection.class.charAt(0).toUpperCase(), x + 8, y + 16);
        });

      } catch (error) {
        console.error('Detection error:', error);
      }

      // Calculate FPS
      frameCount++;
      if (currentTime - lastTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastTime = currentTime;
      }

      // Continue loop for real-time updates
      if (isDetecting) {
        animationRef.current = requestAnimationFrame(detect);
      }
    };

    detect();
  }, [detections, speakDetection, isDetecting]);

  // Handle canvas positioning when video dimensions change
  useEffect(() => {
    const updateCanvasPosition = () => {
      if (videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        
        // Get video display dimensions
        const videoRect = video.getBoundingClientRect();
        
        // Update canvas CSS to match video display size exactly
        canvas.style.width = videoRect.width + 'px';
        canvas.style.height = videoRect.height + 'px';
        canvas.style.top = '0px';
        canvas.style.left = '0px';
        
        console.log('Canvas positioned:', {
          videoRect: { width: videoRect.width, height: videoRect.height },
          canvasStyle: { width: canvas.style.width, height: canvas.style.height },
          canvasSize: { width: canvas.width, height: canvas.height }
        });
      }
    };

    // Update position when video loads
    if (videoRef.current) {
      videoRef.current.addEventListener('loadedmetadata', updateCanvasPosition);
      videoRef.current.addEventListener('resize', updateCanvasPosition);
    }

    // Initial update
    updateCanvasPosition();

    // Cleanup
    return () => {
      if (videoRef.current) {
        videoRef.current.removeEventListener('loadedmetadata', updateCanvasPosition);
        videoRef.current.removeEventListener('resize', updateCanvasPosition);
      }
    };
  }, [isDetecting]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="p-8 text-center">
          <div className="w-12 h-12 mx-auto mb-4 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          <h2 className="text-xl font-semibold mb-2">Loading Detection Model</h2>
          <p className="text-muted-foreground">Initializing TensorFlow.js and COCO-SSD model...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Camera Controls */}
      <div className="p-4 bg-card border-b border-border">
        <div className="flex gap-2 flex-wrap">
          {!isDetecting ? (
            <Button onClick={startCamera} disabled={!modelLoaded} className="gap-2">
              <Camera className="w-4 h-4" />
              Start Detection
            </Button>
          ) : (
            <Button onClick={stopCamera} variant="destructive" className="gap-2">
              <Square className="w-4 h-4" />
              Stop Detection
            </Button>
          )}
          
          <Button
            onClick={() => setSpeechEnabled(!speechEnabled)}
            variant="outline"
            className="gap-2"
            disabled={!window.speechSynthesis}
          >
            {speechEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
            {speechEnabled ? 'Audio On' : 'Audio Off'}
          </Button>
        </div>
      </div>

      {/* Camera View */}
      <div className="relative bg-black">
        <video
          ref={videoRef}
          className="w-full h-auto"
          autoPlay
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-auto z-10"
          style={{ 
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'auto',
            zIndex: 10
          }}
        />
        
        {/* Status Panel */}
        {isDetecting && (
          <div className="absolute top-2 left-2 bg-black bg-opacity-75 text-white p-3 rounded text-sm z-20">
            <div className="font-bold mb-2 border-b border-white/20 pb-1">📊 Detection Status</div>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span>Objects:</span>
                <span className="text-green-400 font-bold">{detections.length}</span>
              </div>
              <div className="flex justify-between">
                <span>FPS:</span>
                <span className={fps >= 30 ? 'text-green-400' : fps >= 25 ? 'text-yellow-400' : 'text-red-400'}>
                  {fps}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Canvas:</span>
                <span className="text-blue-400">{canvasRef.current?.width || 0} x {canvasRef.current?.height || 0}</span>
              </div>
              <div className="flex justify-between">
                <span>Video:</span>
                <span className="text-purple-400">{videoRef.current?.videoWidth || 0} x {videoRef.current?.videoHeight || 0}</span>
              </div>
            </div>
            
            {/* Detection Status */}
            {detections.length > 0 ? (
              <div className="mt-2 pt-2 border-t border-green-400 border-opacity-50">
                <div className="text-xs text-green-300 font-bold">✅ Objects Detected!</div>
                <div className="text-xs text-green-300">
                  Look for colored boxes around objects
                </div>
                <div className="text-xs text-green-300">
                  {detections.map(d => `${d.class} (${d.distance}m)`).join(', ')}
                </div>
              </div>
            ) : (
              <div className="mt-2 pt-2 border-t border-yellow-400 border-opacity-50">
                <div className="text-xs text-yellow-300">🔍 No Objects Detected</div>
                <div className="text-xs text-yellow-300">
                  Move objects in front of camera
                </div>
              </div>
            )}
            
            {/* Distance Legend */}
            <div className="mt-2 pt-2 border-t border-white/20">
              <div className="text-xs font-bold mb-1">Distance Colors:</div>
              <div className="text-xs space-y-1">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span>Red: &lt;3m (Close)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <span>Yellow: 3-5m (Medium)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span>Green: &gt;5m (Far)</span>
                </div>
              </div>
            </div>
            
            {/* Canvas Status */}
            <div className="mt-2 pt-2 border-t border-white/20">
              <div className="text-xs font-bold mb-1">Canvas Status:</div>
              <div className="text-xs text-green-300">
                ✓ Overlay active
              </div>
              <div className="text-xs text-blue-300">
                Look for green "CANVAS OK" box
              </div>
              <div className="text-xs text-blue-300">
                Grid lines for reference
              </div>
            </div>
            
            {/* Enhanced Highlighting Features */}
            <div className="mt-2 pt-2 border-t border-white/20">
              <div className="text-xs font-bold mb-1">🎨 Enhanced Features:</div>
              <div className="text-xs space-y-1">
                <div className="text-green-300">✓ Colored rectangles OVER objects</div>
                <div className="text-green-300">✓ Thick bounding boxes (8px)</div>
                <div className="text-green-300">✓ White borders & corner indicators</div>
                <div className="text-green-300">✓ Center cross markers</div>
                <div className="text-green-300">✓ Object type squares</div>
                <div className="text-green-300">✓ Distance dots with numbers</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ObjectDetectionCamera;