import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Camera, Square, AlertTriangle, Target } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface Detection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
  distance?: number;
  direction?: 'left' | 'center' | 'right';
}

interface Alert {
  id: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  timestamp: number;
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
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [fps, setFps] = useState(0);
  const [modelLoaded, setModelLoaded] = useState(false);

  const { toast } = useToast();

  // Initialize TensorFlow.js and load model
  useEffect(() => {
    const initializeTensorFlow = async () => {
      try {
        await tf.ready();
        console.log('TensorFlow.js initialized');
        
        const model = await cocoSsd.load();
        modelRef.current = model;
        setModelLoaded(true);
        setIsLoading(false);
        
        toast({
          title: "Model Loaded",
          description: "Object detection model ready for use",
        });
      } catch (error) {
        console.error('Error loading model:', error);
        toast({
          variant: "destructive",
          title: "Model Load Error",
          description: "Failed to load object detection model",
        });
      }
    };

    initializeTensorFlow();
  }, [toast]);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play();
            setIsDetecting(true);
            startDetection();
          }
        };
      }
    } catch (error) {
      console.error('Camera access error:', error);
      toast({
        variant: "destructive",
        title: "Camera Access",
        description: "Unable to access camera. Please check permissions.",
      });
    }
  }, [toast]);

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

  // Estimate distance based on bounding box size
  const estimateDistance = (bbox: [number, number, number, number], className: string): number => {
    const [x, y, width, height] = bbox;
    
    // Rough estimates based on average object sizes (in meters)
    const averageHeights: { [key: string]: number } = {
      'person': 1.7,
      'car': 1.5,
      'bicycle': 1.1,
      'motorcycle': 1.2,
      'bus': 3.0,
      'truck': 3.5,
      'dog': 0.6,
      'cat': 0.3,
    };

    const objectHeight = averageHeights[className] || 1.0;
    const focalLength = 800; // Approximate focal length for mobile camera
    
    // Distance = (Real Height × Focal Length) / Pixel Height
    const distance = (objectHeight * focalLength) / height;
    return Math.max(1, Math.min(200, distance)); // Clamp between 1-200m
  };

  // Determine direction based on bounding box position
  const getDirection = (bbox: [number, number, number, number], canvasWidth: number): 'left' | 'center' | 'right' => {
    const [x, y, width] = bbox;
    const centerX = x + width / 2;
    const threshold = canvasWidth * 0.33;
    
    if (centerX < threshold) return 'left';
    if (centerX > canvasWidth - threshold) return 'right';
    return 'center';
  };

  // Generate spatial alerts
  const generateAlert = (detection: Detection): Alert | null => {
    const { class: className, distance, direction, score } = detection;
    
    if (score < 0.5) return null;
    
    let alertType: 'warning' | 'danger' | 'info' = 'info';
    let message = '';

    if (distance && distance < 10) {
      alertType = 'danger';
      message = `${className} very close on the ${direction} (~${Math.round(distance)}m)`;
    } else if (distance && distance < 25) {
      alertType = 'warning';
      message = `${className} approaching from ${direction} (~${Math.round(distance)}m)`;
    } else {
      message = `${className} detected on the ${direction} (~${Math.round(distance || 50)}m)`;
    }

    return {
      id: `${className}-${Date.now()}`,
      message,
      type: alertType,
      timestamp: Date.now()
    };
  };

  // Main detection loop
  const startDetection = useCallback(() => {
    let lastTime = Date.now();
    let frameCount = 0;

    const detect = async () => {
      if (!videoRef.current || !canvasRef.current || !modelRef.current || !isDetecting) {
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      if (!ctx || video.videoWidth === 0) {
        animationRef.current = requestAnimationFrame(detect);
        return;
      }

      // Set canvas size to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Clear canvas and draw video frame
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      try {
        // Run object detection
        const predictions = await modelRef.current.detect(video);
        
        const currentDetections: Detection[] = predictions.map(prediction => {
          const distance = estimateDistance(prediction.bbox, prediction.class);
          const direction = getDirection(prediction.bbox, canvas.width);
          
          return {
            bbox: prediction.bbox,
            class: prediction.class,
            score: prediction.score,
            distance,
            direction
          };
        });

        setDetections(currentDetections);

        // Generate alerts for high-confidence detections
        const newAlerts: Alert[] = [];
        currentDetections.forEach(detection => {
          const alert = generateAlert(detection);
          if (alert) {
            newAlerts.push(alert);
          }
        });

        if (newAlerts.length > 0) {
          setAlerts(prev => [...newAlerts, ...prev].slice(0, 5)); // Keep last 5 alerts
        }

        // Draw bounding boxes and labels
        ctx.strokeStyle = 'hsl(var(--detection-success))';
        ctx.fillStyle = 'hsl(var(--detection-success))';
        ctx.font = '16px Inter, system-ui, sans-serif';
        ctx.lineWidth = 2;

        currentDetections.forEach(detection => {
          const [x, y, width, height] = detection.bbox;
          
          // Draw bounding box
          ctx.strokeRect(x, y, width, height);
          
          // Draw label background
          const label = `${detection.class} ${Math.round(detection.score * 100)}%`;
          const labelWidth = ctx.measureText(label).width + 10;
          ctx.fillStyle = 'hsl(var(--detection-success))';
          ctx.fillRect(x, y - 25, labelWidth, 25);
          
          // Draw label text
          ctx.fillStyle = 'hsl(var(--detection-success-foreground))';
          ctx.fillText(label, x + 5, y - 8);
          
          // Draw distance and direction
          if (detection.distance && detection.direction) {
            const spatialInfo = `${detection.direction} ~${Math.round(detection.distance)}m`;
            ctx.fillStyle = 'hsl(var(--detection-info))';
            ctx.fillText(spatialInfo, x + 5, y + height + 20);
          }
        });

      } catch (error) {
        console.error('Detection error:', error);
      }

      // Calculate FPS
      frameCount++;
      const currentTime = Date.now();
      if (currentTime - lastTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastTime = currentTime;
      }

      animationRef.current = requestAnimationFrame(detect);
    };

    detect();
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
        <Card className="p-8 text-center">
          <Target className="w-12 h-12 mx-auto mb-4 text-primary animate-spin" />
          <h2 className="text-xl font-semibold mb-2">Loading Detection Model</h2>
          <p className="text-muted-foreground">Initializing TensorFlow.js and COCO-SSD model...</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="bg-card border-b border-border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Target className="w-6 h-6 text-primary" />
            <h1 className="text-xl font-semibold">Object Detection</h1>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant="secondary" className="gap-1">
              <Square className="w-3 h-3" />
              {detections.length} objects
            </Badge>
            <Badge variant="outline">{fps} FPS</Badge>
          </div>
        </div>
      </div>

      {/* Camera Controls */}
      <div className="p-4 bg-card border-b border-border">
        <div className="flex gap-2">
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
        </div>
      </div>

      {/* Alert Panel */}
      {alerts.length > 0 && (
        <div className="p-4 bg-card border-b border-border">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="w-4 h-4 text-destructive" />
            <span className="font-medium">Spatial Alerts</span>
          </div>
          <div className="space-y-2">
            {alerts.slice(0, 3).map(alert => (
              <div
                key={alert.id}
                className={`p-2 rounded-lg text-sm ${
                  alert.type === 'danger' 
                    ? 'bg-destructive/10 text-destructive border border-destructive/20'
                    : alert.type === 'warning'
                    ? 'bg-accent/10 text-accent-foreground border border-accent/20'
                    : 'bg-detection-info/10 text-detection-info border border-detection-info/20'
                }`}
              >
                {alert.message}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Camera View */}
      <div className="relative flex-1">
        <video
          ref={videoRef}
          className="w-full h-auto max-h-[60vh] object-cover"
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full object-cover pointer-events-none"
        />
        
        {/* Overlay Info */}
        {isDetecting && (
          <div className="absolute top-4 left-4 bg-card/90 rounded-lg p-3 border border-border backdrop-blur-sm">
            <div className="text-sm space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-detection-success rounded-full animate-pulse" />
                <span>Live Detection Active</span>
              </div>
              <div className="text-muted-foreground">
                Model: COCO-SSD | FPS: {fps}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Detection Stats */}
      {detections.length > 0 && (
        <div className="p-4 bg-card border-t border-border">
          <h3 className="font-medium mb-3">Current Detections</h3>
          <div className="grid grid-cols-1 gap-2">
            {detections.slice(0, 5).map((detection, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-muted rounded-lg">
                <div className="flex items-center gap-2">
                  <Badge variant="secondary">{detection.class}</Badge>
                  <span className="text-sm text-muted-foreground">
                    {Math.round(detection.score * 100)}% confidence
                  </span>
                </div>
                <div className="text-sm">
                  <span className="text-detection-info">
                    {detection.direction} • {Math.round(detection.distance || 0)}m
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ObjectDetectionCamera;