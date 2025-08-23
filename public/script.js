// Mobile-Friendly Object Detection with Distance Estimation, Directional Awareness, and Voice Alerts
// Using TensorFlow.js COCO-SSD model

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');

// State Variables
let isDetecting = false;
let model = null;
let stream = null;
let animationId = null;
let voiceEnabled = true;
let detections = [];
let frameCount = 0;
let lastTime = performance.now();
let lastDetectionTime = 0;

// Constants
const DETECTION_INTERVAL = 100; // ms
const FOCAL_LENGTH = 1000; // pixels
const KNOWN_HEIGHTS = {
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
  'bird': 0.15,       // 0.15 meters average height
  'horse': 1.6,       // 1.6 meters average height
  'sheep': 1.0,       // 1.0 meters average height
  'cow': 1.4,         // 1.4 meters average height
  'elephant': 3.0,    // 3.0 meters average height
  'bear': 1.8,        // 1.8 meters average height
  'zebra': 1.0,       // 1.0 meters average height
  'giraffe': 4.0,     // 4.0 meters average height
  'backpack': 0.5,    // 0.5 meters average height
};

// Distance estimation using focal length method
function estimateDistance(bbox, className) {
  const [x, y, width, height] = bbox;
  const knownHeight = KNOWN_HEIGHTS[className] || 1.0; // Default to 1 meter if unknown
  
  // Distance = (Known Height × Focal Length) / Perceived Height
  const distance = (knownHeight * FOCAL_LENGTH) / height;
  
  // Apply mobile camera correction factor (typically 1.2-1.5x)
  const correctedDistance = distance * 1.3;
  
  return Math.max(0.5, Math.min(50, correctedDistance)); // Clamp between 0.5m and 50m
}

// Get direction based on object position
function getDirection(x, width, videoWidth) {
  const centerX = x + width / 2;
  if (centerX < videoWidth / 3) return "Left";
  if (centerX > (2 * videoWidth) / 3) return "Right";
  return "Center";
}

// Get color based on distance
function getDistanceColor(distance) {
  if (distance < 3) return "#FF0000";      // Red: < 3m
  if (distance < 5) return "#FFFF00";      // Yellow: 3-5m
  return "#00FF00";                        // Green: > 5m
}

// Voice synthesis with cooldown
function speak(message) {
  if (!voiceEnabled) return;
  
  const synth = window.speechSynthesis;
  if (!synth.speaking) {
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.lang = "en-US";
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    utterance.volume = 0.8;
    synth.speak(utterance);
    console.log('Voice alert:', message);
  }
}

// Generate voice alert message
function generateAlertMessage(detection) {
  const { class: className, distance, direction } = detection;
  
  if (distance < 3) {
    return `${className} approaching within ${distance.toFixed(1)} meters on the ${direction}`;
  } else if (distance < 5) {
    return `${className} detected ${distance.toFixed(1)} meters away on the ${direction}`;
  } else {
    return `${className} detected about ${distance.toFixed(1)} meters in the ${direction}`;
  }
}

// Start camera
async function startCamera() {
  try {
    console.log('Starting camera...');
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'environment', // Use back camera on mobile
        width: { ideal: 640, max: 1280 },
        height: { ideal: 480, max: 720 },
        frameRate: { ideal: 30, max: 30 }
      }
    });
    
    video.srcObject = stream;
    
    video.onloadedmetadata = () => {
      console.log('Video ready, starting detection...');
      video.play();
      startDetection();
    };
  } catch (error) {
    console.error('Camera access error:', error);
    statusDiv.textContent = 'Error: Camera access denied';
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
  updateStatus();
}

// Load COCO-SSD model
async function loadModel() {
  try {
    console.log('Loading COCO-SSD model...');
    model = await cocoSsd.load();
    console.log('COCO-SSD model loaded successfully');
    updateStatus();
  } catch (error) {
    console.error('Error loading model:', error);
    statusDiv.textContent = 'Error: Failed to load model';
  }
}

// Main detection loop
function startDetection() {
  isDetecting = true;
  updateStatus();
  
  const detect = async (currentTime) => {
    if (!video || !canvas || !model || !isDetecting) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    // Set canvas dimensions to match video
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

    // Run object detection at controlled intervals
    if (currentTime - lastDetectionTime >= DETECTION_INTERVAL) {
      try {
        console.log('Running object detection...');
        
        const predictions = await model.detect(video);
        const filteredPredictions = predictions.filter(prediction => prediction.score > 0.5);
        
        if (filteredPredictions.length > 0) {
          console.log('Objects detected:', filteredPredictions.map(p => `${p.class}:${Math.round(p.score*100)}%`));
        }
        
        // Process detections with distance estimation and direction
        detections = filteredPredictions.map(prediction => {
          const [x, y, width, height] = prediction.bbox;
          const distance = estimateDistance(prediction.bbox, prediction.class);
          const direction = getDirection(x, width, video.videoWidth);
          
          return {
            id: `${prediction.class}-${Date.now()}-${Math.random()}`,
            bbox: prediction.bbox,
            class: prediction.class,
            score: prediction.score,
            distance,
            direction,
            lastAlertTime: 0
          };
        });
        
        lastDetectionTime = currentTime;
        
        // Draw bounding boxes and labels
        detections.forEach((detection, index) => {
          const [x, y, width, height] = detection.bbox;
          const color = getDistanceColor(detection.distance);
          
          // Validate bbox coordinates
          if (x < 0 || y < 0 || x + width > canvas.width || y + height > canvas.height) {
            console.warn('Bbox coordinates out of bounds:', { x, y, width, height, canvasWidth: canvas.width, canvasHeight: canvas.height });
            return;
          }
          
          // Draw bounding box
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, width, height);
          
          // Draw label background
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
          
          // Voice alerts with cooldown
          const now = Date.now();
          if (detection.distance < 5 && now - detection.lastAlertTime > 5000) {
            const alertMessage = generateAlertMessage(detection);
            speak(alertMessage);
            detection.lastAlertTime = now;
          }
          
          console.log(`Drew object ${index}: ${detection.class} at ${detection.distance.toFixed(1)}m ${detection.direction}`);
        });
        
      } catch (error) {
        console.error('Detection error:', error);
      }
    }

    // Calculate FPS
    frameCount++;
    if (currentTime - lastTime >= 1000) {
      const fps = frameCount;
      frameCount = 0;
      lastTime = currentTime;
      updateStatus(fps);
    }

    // Continue loop for real-time updates
    if (isDetecting) {
      animationId = requestAnimationFrame(detect);
    }
  };

  detect(performance.now());
}

// Update status display
function updateStatus(fps = 0) {
  const fpsText = fps > 0 ? ` | FPS: ${fps}` : '';
  const objectsText = ` | Objects: ${detections.length}`;
  const modelText = ` | Model: ${model ? 'Ready' : 'Loading'}`;
  
  statusDiv.textContent = `Status: ${isDetecting ? 'Detecting' : 'Stopped'}${fpsText}${objectsText}${modelText}`;
}

// Event listeners
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

// Initialize
loadModel();
updateStatus();