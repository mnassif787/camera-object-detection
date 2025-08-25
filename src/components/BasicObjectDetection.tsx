import React, { useRef, useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Camera, Square, Volume2, VolumeX } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import useVoiceAlerts from '@/hooks/useVoiceAlerts';
import SettingsPanel from './SettingsPanel';

interface Detection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
  distance: number;
  direction: string;
}

const BasicObjectDetection: React.FC = () => {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);

  // State
  const [isLoading, setIsLoading] = useState(true);
  const [isDetecting, setIsDetecting] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);

  // Voice Alerts
  const {
    isSupported: isVoiceSupported,
    isSpeaking,
    options: voiceAlertOptions,
    initializeSpeech,
    announceDetection,
    announceSummary,
    stopSpeaking,
    updateOptions: updateVoiceOptions,
    testVoice,
  } = useVoiceAlerts();

  // Constants for distance estimation
  const FOCAL_LENGTH = 1000; // pixels
  const KNOWN_HEIGHTS: { [key: string]: number } = {
    'person': 1.7,      // 1.7 meters average height
    'car': 1.5,         // 1.5 meters average height
    'bicycle': 1.5,     // 1.5 meters average height
    'motorcycle': 1.5,  // 1.5 meters average height
    'bus': 3.0,         // 3.0 meters average height
    'truck': 3.0,       // 3.0 meters average height
    'dog': 0.5,         // 0.5 meters average height
    'cat': 0.3,         // 0.3 meters average height
    'chair': 0.9,       // 0.9 meters average height
    'table': 0.8,       // 0.8 meters average height
    'bottle': 0.25,     // 0.25 meters average height
    'cup': 0.15,        // 0.15 meters average height
    'laptop': 0.3,      // 0.3 meters average height
    'tv': 0.6,          // 0.6 meters average height
    'cell phone': 0.15, // 0.15 meters average height
    'book': 0.25,       // 0.25 meters average height
    'bowl': 0.1,        // 0.1 meters average height
    'apple': 0.08,      // 0.08 meters average height
    'banana': 0.2,      // 0.2 meters average height
    'airplane': 10.0,   // 10.0 meters average height
    'train': 4.0,       // 4.0 meters average height
    'boat': 2.0,        // 2.0 meters average height
    'fire hydrant': 0.8, // 0.8 meters average height
    'stop sign': 0.8,   // 0.8 meters average height
    'bench': 0.5,       // 0.5 meters average height
    'bird': 0.2,        // 0.2 meters average height
    'horse': 1.6,       // 1.6 meters average height
    'sheep': 1.0,       // 1.0 meters average height
    'cow': 1.4,         // 1.4 meters average height
    'elephant': 3.0,    // 3.0 meters average height
    'bear': 1.8,        // 1.8 meters average height
    'zebra': 1.4,       // 1.4 meters average height
    'giraffe': 4.0,     // 4.0 meters average height
    'backpack': 0.5,    // 0.5 meters average height
  };

  // Load COCO-SSD model and initialize voice alerts
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log('Loading COCO-SSD model...');
        
        // Wait for TensorFlow.js to be ready
        await tf.ready();
        console.log('TensorFlow.js ready');
        
        // Load the model
        const model = await cocoSsd.load();
        modelRef.current = model;
        setModelLoaded(true);
        setIsLoading(false);
        console.log('COCO-SSD model loaded successfully');
      } catch (error) {
        console.error('Error loading model:', error);
        setIsLoading(false);
      }
    };

    loadModel();
    initializeSpeech();
  }, [initializeSpeech]);

  // Distance estimation using focal length method
  const estimateDistance = (bbox: [number, number, number, number], className: string): number => {
    const [x, y, width, height] = bbox;
    const knownHeight = KNOWN_HEIGHTS[className] || 1.0; // Default to 1 meter if unknown
    
    // Distance = (Known Height × Focal Length) / Perceived Height
    const distance = (knownHeight * FOCAL_LENGTH) / height;
    
    // Apply mobile camera correction factor (typically 1.2-1.5x)
    const correctedDistance = distance * 1.3;
    
    // Clamp between 0.5m and 50m for realistic values
    return Math.max(0.5, Math.min(50, correctedDistance));
  };

  // Get direction based on object position
  const getDirection = (x: number, width: number, videoWidth: number): string => {
    const centerX = x + width / 2;
    if (centerX < videoWidth / 3) return "Left";
    if (centerX > (2 * videoWidth) / 3) return "Right";
    return "Center";
  };

  // Get color based on distance
  const getDistanceColor = (distance: number): string => {
    if (distance < 3) return "#FF0000";      // Red: < 3m
    if (distance < 5) return "#FFFF00";      // Yellow: 3-5m
    return "#00FF00";                        // Green: > 5m
  };

  // Start camera
  const startCamera = async () => {
    try {
      console.log('Starting camera...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Use back camera on mobile
          width: { ideal: 640, max: 1280 },
          height: { ideal: 480, max: 720 }
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        
        videoRef.current.onloadedmetadata = () => {
          console.log('Video ready, starting detection...');
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
  };

  // Stop camera
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsDetecting(false);
    setDetections([]);
  };

  // Main detection loop
  const startDetection = () => {
    let frameCount = 0;
    let lastTime = performance.now();

    const detect = async () => {
      if (!videoRef.current || !canvasRef.current || !modelRef.current || !isDetecting) {
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      if (!ctx) return;

      // Set canvas dimensions to match video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      try {
        // Run object detection
        const predictions = await modelRef.current.detect(video);
        const filteredPredictions = predictions.filter(prediction => prediction.score > confidenceThreshold);
        
        // Process detections with distance estimation
        const newDetections: Detection[] = filteredPredictions.map(prediction => {
          const distance = estimateDistance(prediction.bbox, prediction.class);
          const direction = getDirection(prediction.bbox[0], prediction.bbox[2], video.videoWidth);
          return {
            bbox: prediction.bbox,
            class: prediction.class,
            score: prediction.score,
            distance,
            direction
          };
        });
        
        // Update detections state
        setDetections(newDetections);
        
        // Announce detections via voice alerts
        newDetections.forEach(detection => {
          announceDetection(detection);
        });
        
        // Draw bounding boxes with distance-based colors
        newDetections.forEach(detection => {
          const [x, y, width, height] = detection.bbox;
          const color = getDistanceColor(detection.distance);
          
          // Draw bounding box
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, width, height);
          
          // Draw label with distance and direction
          const label = `${detection.class} (${detection.distance.toFixed(1)}m, ${detection.direction})`;
          const labelPadding = 8;
          const labelHeight = 20;
          const labelWidth = ctx.measureText(label).width + labelPadding * 2;
          
          // Position label above the object, or below if too close to top
          const labelY = y > labelHeight + 10 ? y - 10 : y + height + 10;
          const labelX = x;
          
          // Label background
          ctx.fillStyle = color + 'CC'; // Semi-transparent
          ctx.fillRect(labelX, labelY - labelHeight + 5, labelWidth, labelHeight);
          
          // Label border
          ctx.strokeStyle = color;
          ctx.lineWidth = 1;
          ctx.strokeRect(labelX, labelY - labelHeight + 5, labelWidth, labelHeight);
          
          // Label text
          ctx.fillStyle = '#FFFFFF';
          ctx.font = 'bold 14px Arial';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(label, labelX + labelPadding, labelY - labelHeight/2 + 5);
        });
        
      } catch (error) {
        console.error('Detection error:', error);
      }

      // Calculate FPS
      frameCount++;
      const currentTime = performance.now();
      if (currentTime - lastTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastTime = currentTime;
        
        // Periodic summary announcement (every 10 seconds)
        if (frameCount % 10 === 0 && (voiceAlertOptions.alertType === 'summary' || voiceAlertOptions.alertType === 'both')) {
          announceSummary(newDetections);
        }
      }

      // Continue loop
      if (isDetecting) {
        requestAnimationFrame(detect);
      }
    };

    // Start the detection loop
    detect();
    
    // Announce summary if enabled
    if (voiceAlertOptions.alertType === 'summary' || voiceAlertOptions.alertType === 'both') {
      // Delay summary to avoid conflicts with immediate announcements
      setTimeout(() => {
        announceSummary(detections);
      }, 2000);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      stopSpeaking();
    };
  }, [stopSpeaking]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="p-8 text-center">
          <div className="w-12 h-12 mx-auto mb-4 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          <h2 className="text-xl font-semibold mb-2">Loading Detection Model</h2>
          <p className="text-muted-foreground">Please wait while we load the AI model...</p>
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
          
          {/* Status Info */}
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span>FPS: {fps}</span>
            <span>Objects: {detections.length}</span>
            <span>Model: {modelLoaded ? "Ready" : "Loading"}</span>
            <span>Backend: {tf.getBackend()}</span>
            <span className="flex items-center gap-1">
              {isVoiceSupported ? (
                voiceAlertOptions.enabled ? (
                  <Volume2 className="w-4 h-4 text-green-500" />
                ) : (
                  <VolumeX className="w-4 h-4 text-muted-foreground" />
                )
              ) : (
                <VolumeX className="w-4 h-4 text-red-500" />
              )}
              Voice
            </span>
          </div>
        </div>

        {/* Confidence Threshold Slider */}
        <div className="mt-4">
          <label className="text-sm font-medium mb-2 block">
            Confidence Threshold: {(confidenceThreshold * 100).toFixed(0)}%
          </label>
          <Slider
            value={[confidenceThreshold]}
            onValueChange={(value) => setConfidenceThreshold(value[0])}
            max={1}
            min={0.1}
            step={0.05}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>10% (More Objects)</span>
            <span>100% (Fewer Objects)</span>
          </div>
        </div>

        {/* Settings Panel */}
        <div className="mt-4">
          <SettingsPanel
            voiceAlertOptions={voiceAlertOptions}
            onVoiceAlertOptionsChange={updateVoiceOptions}
            onTestVoice={testVoice}
            isSpeaking={isSpeaking}
            isVoiceSupported={isVoiceSupported}
          />
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
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>

      {/* Direction Zones Info */}
      <div className="p-4 bg-card border-t border-border">
        <h3 className="font-semibold mb-2">Direction Zones</h3>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="text-center p-2 bg-red-50 border border-red-200 rounded">
            <div className="font-bold text-red-700">Left</div>
            <div className="text-red-600">0-33%</div>
          </div>
          <div className="text-center p-2 bg-yellow-50 border border-yellow-200 rounded">
            <div className="font-bold text-yellow-700">Center</div>
            <div className="text-yellow-600">34-66%</div>
          </div>
          <div className="text-center p-2 bg-green-50 border border-green-200 rounded">
            <div className="font-bold text-green-700">Right</div>
            <div className="text-green-600">67-100%</div>
          </div>
        </div>
      </div>

      {/* Distance Legend */}
      <div className="p-4 bg-card border-t border-border">
        <h3 className="font-semibold mb-2">Distance Colors</h3>
        <div className="flex gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span>Red: &lt;3m (Close)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-500 rounded"></div>
            <span>Yellow: 3-5m (Medium)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 rounded"></div>
            <span>Green: &gt;5m (Far)</span>
          </div>
        </div>
      </div>

      {/* Detection Info */}
      <div className="p-4 bg-card border-t border-border">
        <h3 className="font-semibold mb-2">Detected Objects</h3>
        {detections.length === 0 ? (
          <p className="text-muted-foreground">No objects detected yet. Start the camera to begin detection.</p>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            {detections.map((detection, index) => (
              <div key={index} className="p-2 bg-muted rounded text-sm">
                <div className="font-medium">{detection.class}</div>
                <div className="text-muted-foreground">
                  {Math.round(detection.score * 100)}% • {detection.distance.toFixed(1)}m • {detection.direction}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default BasicObjectDetection;