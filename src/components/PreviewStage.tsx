import { Play, Upload, Image as ImageIcon, Mic, Type } from "lucide-react";
import { useRef } from "react";
import { Button } from "@/components/ui/button";

interface PreviewStageProps {
  hasImage: boolean;
  hasVideo: boolean;
  isGenerating: boolean;
  onUploadImage: () => void;
  selectedImage?: File | null;
  selectedAudio?: File | null;
  textInput: string;
  onTextChange: (text: string) => void;
  currentPlayingVideo?: string | null;
  filmstripItems?: FilmstripItem[];
  logs?: string[];
}

interface FilmstripItem {
  id: string;
  title: string;
  status: 'ready' | 'processing' | 'error';
  videoUrl?: string;
  thumbnail?: string;
}

export const PreviewStage = ({ 
  hasImage, 
  hasVideo, 
  isGenerating, 
  onUploadImage,
  selectedImage,
  selectedAudio,
  textInput,
  onTextChange,
  currentPlayingVideo,
  filmstripItems,
  logs
}: PreviewStageProps) => {
  const logRef = useRef<HTMLDivElement | null>(null);
  return (
    <div className="glass-card p-8 aspect-video flex flex-col items-center justify-center relative overflow-hidden preview-stage max-w-4xl mx-auto">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-5 dark:opacity-10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--primary)_1px,_transparent_1px)] bg-[size:40px_40px]"></div>
      </div>
      
      {!hasImage && !isGenerating && (
        <div className="text-center space-y-6 relative z-10 w-full max-w-md mx-auto">
          {/* Appropriately sized upload icon */}
          <div className="w-28 h-28 mx-auto glass-card rounded-full flex items-center justify-center animate-float shadow-lg">
            <ImageIcon size={56} className="text-primary" />
          </div>
          
          {/* Upload section with proper sizing */}
          <div className="space-y-4">
            <h2 className="text-2xl font-bold gradient-text">
              Upload Image to Start
            </h2>
            <p className="text-muted-foreground text-base leading-relaxed">
              Choose a photo to transform into a talking video
            </p>
            
            {/* Properly sized upload button */}
            <div className="pt-4">
              <Button 
                onClick={onUploadImage}
                className="glass-button text-white text-base px-8 py-3 h-auto shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105"
                size="default"
              >
                <Upload className="mr-2" size={20} />
                Choose Image
              </Button>
            </div>
            
            {/* Helper text */}
            <p className="text-xs text-muted-foreground/70 mt-4">
              Supported formats: JPG, PNG, WEBP
            </p>
          </div>
        </div>
      )}

      {hasImage && !hasVideo && !isGenerating && (
        <div className="w-full h-full flex flex-col space-y-4 relative z-10">
          {/* Image Preview */}
          <div className="flex-1 flex items-center justify-center">
            {selectedImage && (
              <div className="w-32 h-32 glass-card rounded-2xl overflow-hidden">
                <img 
                  src={URL.createObjectURL(selectedImage)} 
                  alt="Selected" 
                  className="w-full h-full object-cover"
                />
              </div>
            )}
          </div>
          
          {/* Input Controls */}
          <div className="space-y-4">
            {/* Audio Input */}
            <div className="flex items-center gap-3 p-3 glass-card rounded-lg">
              <Mic size={20} className="text-primary" />
              <div className="flex-1">
                <p className="text-sm font-medium">Audio Input</p>
                <p className="text-xs text-muted-foreground">
                  {selectedAudio ? selectedAudio.name : 'No audio selected'}
                </p>
              </div>
            </div>
            
            {/* Text Input */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Type size={16} className="text-primary" />
                <label className="text-sm font-medium">Script Text</label>
              </div>
              <textarea
                value={textInput}
                onChange={(e) => onTextChange(e.target.value)}
                placeholder="Enter the text you want the talking head to speak..."
                className="w-full p-3 bg-muted/20 border border-border rounded-lg text-sm resize-none dark:bg-gray-700/20 dark:border-gray-600"
                rows={3}
              />
            </div>
          </div>
        </div>
      )}

      {isGenerating && (
        <div className="text-center space-y-6 relative z-10 w-full max-w-3xl">
          <div className="relative">
            <div className="w-24 h-24 mx-auto glass-card rounded-full flex items-center justify-center animate-pulse-glow">
              <Play size={48} className="text-white" />
            </div>
            <div className="absolute inset-0 w-24 h-24 mx-auto rounded-full border-4 border-primary/30 border-t-primary animate-spin"></div>
          </div>
          <div>
            <h3 className="text-xl font-bold gradient-text">Generating Your Video...</h3>
            <p className="text-muted-foreground">This might take a few moments</p>
          </div>
          {/* Single line live status */}
          <div className="mt-2 min-h-[1.5rem]">
            <p ref={logRef} className="text-sm text-gray-200/90 font-mono">
              {(() => {
                const items = (logs || []).filter(l => l && !l.startsWith('[logs]'));
                if (items.length === 0) return 'Waiting for logs...';
                return items[items.length - 1];
              })()}
            </p>
          </div>
        </div>
      )}

      {hasVideo && (
        <div className="w-full h-full relative rounded-xl overflow-hidden">
          <div className="w-full h-full bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center">
            <Button className="glass-button text-white">
              <Play className="mr-2" size={20} />
              Play Video
            </Button>
          </div>
        </div>
      )}

      {/* Video Player Display */}
      {currentPlayingVideo && filmstripItems && (
        <div className="w-full h-full flex flex-col items-center justify-center space-y-4">
          <div className="w-full max-w-2xl">
            {(() => {
              const videoItem = filmstripItems.find(item => item.id === currentPlayingVideo);
              if (videoItem && videoItem.videoUrl) {
                return (
                  <video
                    src={videoItem.videoUrl}
                    controls
                    className="w-full rounded-lg shadow-lg"
                    autoPlay
                  >
                    Your browser does not support the video tag.
                  </video>
                );
              }
              return (
                <div className="w-full h-64 bg-muted/20 border border-border rounded-lg flex items-center justify-center dark:bg-gray-700/20 dark:border-gray-600">
                  <span className="text-muted-foreground">Video not found</span>
                </div>
              );
            })()}
          </div>
          <div className="text-center">
            <h3 className="text-lg font-semibold mb-2">Now Playing</h3>
            <p className="text-sm text-muted-foreground">
              {filmstripItems.find(item => item.id === currentPlayingVideo)?.title}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
