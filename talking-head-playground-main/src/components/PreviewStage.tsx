import { Play, Upload, Image as ImageIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

interface PreviewStageProps {
  hasImage: boolean;
  hasVideo: boolean;
  isGenerating: boolean;
  onUploadImage: () => void;
}

export const PreviewStage = ({ 
  hasImage, 
  hasVideo, 
  isGenerating, 
  onUploadImage 
}: PreviewStageProps) => {
  return (
    <div className="glass-card p-8 aspect-video flex flex-col items-center justify-center relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--primary)_1px,_transparent_1px)] bg-[size:40px_40px]"></div>
      </div>
      
      {!hasImage && !isGenerating && (
        <div className="text-center space-y-6 relative z-10">
          <div className="w-24 h-24 mx-auto glass-card rounded-full flex items-center justify-center animate-float">
            <ImageIcon size={48} className="text-primary" />
          </div>
          <div>
            <h2 className="text-2xl font-bold mb-2 gradient-text">
              Upload Image to Start
            </h2>
            <p className="text-muted-foreground mb-6">
              Choose a photo to transform into a talking video
            </p>
            <Button 
              onClick={onUploadImage}
              className="glass-button text-white"
            >
              <Upload className="mr-2" size={20} />
              Choose Image
            </Button>
          </div>
        </div>
      )}

      {hasImage && !hasVideo && !isGenerating && (
        <div className="text-center space-y-4 relative z-10">
          <div className="w-32 h-32 mx-auto glass-card rounded-2xl bg-muted/20 flex items-center justify-center">
            <ImageIcon size={64} className="text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Image Ready!</h3>
            <p className="text-muted-foreground">Add voice and text to generate video</p>
          </div>
        </div>
      )}

      {isGenerating && (
        <div className="text-center space-y-6 relative z-10">
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
    </div>
  );
};