// Standalone Object Detection Script
// Implements real-time object detection with distance-based colored bounding boxes
// Based on TensorFlow.js COCO-SSD model

import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';

// DOM elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// State variables
let isDetecting = false;
let detections = [];
let model = null;
let stream = null;
let animationId = null;

// Known real-world dimensions for common objects (in meters)
const objectDimensions = {
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

// Distance estimation using focal length approximation
function estimateDistance(bbox, className) {
  const [x, y, width, height] = bbox;
  
  // Get known width for this object type, default to 0.5m if unknown
  const knownWidth = objectDimensions[className] || 0.5;
  
  // Focal length approximation (adjust experimentally for your camera)
  const focalLength = 700;
  
  // Calculate distance using focal length formula: distance = (known_width * focal_length) / perceived_width
  const distance = (knownWidth * focalLength) / width;
  
  // Clamp distance to reasonable bounds (0.5m to 50m)
  const clampedDistance = Math.max(0.5, Math.min(50, distance));
  
  return Math.round(clampedDistance * 10) / 10; // Round to 1 decimal place
}

// Get color based on distance (exactly as specified)
function getDistanceColor(distance) {
  if (distance < 3) return '#FF0000';      // Red if distance < 3m
  if (distance <= 5) return '#FFFF00';     // Yellow if distance between 3m and 5m
  return '#00FF00';                        // Green if distance > 5m
}

// Start camera
async function startCamera() {
  try {
    console.log('Starting camera...');
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'environment',
        width: { ideal: 640, max: 1280 },
        height: { ideal: 480, max: 720 },
        frameRate: { ideal: 30, max: 30 }
      }
    });
    
    video.srcObject = stream;
    video.play();
    
    // Wait for video to be ready
    video.onloadedmetadata = () => {
      console.log('Video ready, starting detection...');
      startDetection();
    };
    
  } catch (error) {
    console.error('Camera access error:', error);
  }
}

// Stop camera
function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  if (animationId) {
    cancelAnimationFrame(animationId);
  }
  isDetecting = false;
  detections = [];
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Load COCO-SSD model
async function loadModel() {
  try {
    console.log('Loading COCO-SSD model...');
    model = await cocoSsd.load();
    console.log('Model loaded successfully');
    return true;
  } catch (error) {
    console.error('Error loading model:', error);
    return false;
  }
}

// Main detection loop - optimized for 30+ FPS
function startDetection() {
  if (!model || !video || !canvas) {
    console.error('Model, video, or canvas not ready');
    return;
  }
  
  console.log('Starting detection loop...');
  isDetecting = true;
  
  let lastDetectionTime = 0;
  const detectionInterval = 200; // Run detection every 200ms for smooth 30+ FPS
  
  function detect() {
    if (!isDetecting) return;
    
    const currentTime = Date.now();
    
    // Set canvas size to match video dimensions
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      console.log('Canvas resized to:', canvas.width, 'x', canvas.height);
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Run object detection at controlled intervals
    if (currentTime - lastDetectionTime >= detectionInterval) {
      model.detect(video).then(predictions => {
        // Filter predictions by confidence
        const filteredPredictions = predictions.filter(prediction => prediction.score > 0.3);
        
        if (filteredPredictions.length > 0) {
          console.log('Objects detected:', filteredPredictions.map(p => `${p.class}:${Math.round(p.score*100)}%`));
        }
        
        // Process detections with distance estimation
        detections = filteredPredictions.map(prediction => {
          const distance = estimateDistance(prediction.bbox, prediction.class);
          return {
            bbox: prediction.bbox,
            class: prediction.class,
            score: prediction.score,
            distance
          };
        });
        
        lastDetectionTime = currentTime;
      }).catch(error => {
        console.error('Detection error:', error);
      });
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
      
      // Enhanced Professional Bounding Box Drawing (YOLOv7 Style)
      
      // 1. Semi-transparent filled background for better visibility
      ctx.fillStyle = color + '20'; // Very light color with transparency
      ctx.fillRect(x, y, width, height);
      
      // 2. Main bounding box with professional styling
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);
      
      // 3. Inner highlight border for depth
      ctx.strokeStyle = color + '80';
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 1, y + 1, width - 2, height - 2);
      
      // 4. Corner indicators for professional look
      const cornerSize = 8;
      const cornerColor = color;
      
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
      
      // 5. Professional Label Background
      const label = `${detection.class.toUpperCase()} ${detection.distance.toFixed(1)}m`;
      const labelPadding = 8;
      const labelHeight = 24;
      const labelWidth = ctx.measureText(label).width + labelPadding * 2;
      
      // Position label above the object, or below if too close to top
      const labelY = y > labelHeight + 10 ? y - 10 : y + height + 10;
      const labelX = x;
      
      // Label background with rounded corners effect
      ctx.fillStyle = color + 'F0'; // Solid color with slight transparency
      ctx.fillRect(labelX, labelY - labelHeight + 5, labelWidth, labelHeight);
      
      // Label border
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(labelX, labelY - labelHeight + 5, labelWidth, labelHeight);
      
      // 6. Professional Text Rendering
      ctx.fillStyle = '#FFFFFF';
      ctx.font = 'bold 14px Arial';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, labelX + labelPadding, labelY - labelHeight/2 + 5);
      
      // 7. Confidence Score Indicator (Professional Style)
      const confidence = Math.round(detection.score * 100);
      const confidenceColor = confidence > 80 ? '#00FF00' : confidence > 60 ? '#FFFF00' : '#FF0000';
      
      // Confidence circle
      ctx.fillStyle = confidenceColor;
      ctx.beginPath();
      ctx.arc(x + width - 15, y + 15, 8, 0, 2 * Math.PI);
      ctx.fill();
      
      // Confidence border
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Confidence percentage
      ctx.fillStyle = '#FFFFFF';
      ctx.font = 'bold 10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(confidence.toString(), x + width - 15, y + 15);
      
      // 8. Distance Indicator (Professional Style)
      const distanceDotSize = 6;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x + 15, y + 15, distanceDotSize, 0, 2 * Math.PI);
      ctx.fill();
      
      // Distance dot border
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // 9. Object Type Icon (Professional Touch)
      const iconSize = 16;
      const iconX = x + width/2;
      const iconY = y + height/2;
      
      // Icon background
      ctx.fillStyle = color + '80';
      ctx.fillRect(iconX - iconSize/2, iconY - iconSize/2, iconSize, iconSize);
      
      // Icon border
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.strokeRect(iconX - iconSize/2, iconY - iconSize/2, iconSize, iconSize);
      
      // Icon text (first letter of object class)
      ctx.fillStyle = '#FFFFFF';
      ctx.font = 'bold 12px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(detection.class.charAt(0).toUpperCase(), iconX, iconY);
    });
    
    // Continue loop for real-time updates
    if (isDetecting) {
      animationId = requestAnimationFrame(detect);
    }
  }
  
  detect();
}

// Initialize the application
async function init() {
  console.log('Initializing object detection app...');
  
  // Load model first
  const modelLoaded = await loadModel();
  if (!modelLoaded) {
    console.error('Failed to load model');
    return;
  }
  
  // Add event listeners for start/stop buttons
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  
  if (startBtn) {
    startBtn.addEventListener('click', startCamera);
  }
  
  if (stopBtn) {
    stopBtn.addEventListener('click', stopCamera);
  }
  
  console.log('App initialized successfully');
}

// Export functions for use in other scripts
window.ObjectDetection = {
  init,
  startCamera,
  stopCamera,
  estimateDistance,
  getDistanceColor
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}