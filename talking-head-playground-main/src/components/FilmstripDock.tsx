import { Play, Download, MoreHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FilmstripItem {
  id: string;
  title: string;
  status: 'processing' | 'ready' | 'failed';
  thumbnail?: string;
}

interface FilmstripDockProps {
  items: FilmstripItem[];
  onPlayItem: (id: string) => void;
  onDownloadItem: (id: string) => void;
}

export const FilmstripDock = ({ items, onPlayItem, onDownloadItem }: FilmstripDockProps) => {
  return (
    <div className="glass-card p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-sm">Video Takes</h3>
        <Button variant="ghost" size="sm">
          <MoreHorizontal size={16} />
        </Button>
      </div>
      
      <div className="flex gap-3 overflow-x-auto pb-2">
        {items.map((item, index) => (
          <div key={item.id} className="flex-shrink-0 space-y-2">
            <div 
              className={`filmstrip-item ${item.status === 'processing' ? 'processing' : ''}`}
            >
              {item.status === 'ready' && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full h-full"
                  onClick={() => onPlayItem(item.id)}
                >
                  <Play size={16} />
                </Button>
              )}
              {item.status === 'processing' && (
                <div className="text-xs text-center">
                  <div className="w-4 h-4 mx-auto mb-1 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  Processing
                </div>
              )}
              {item.status === 'failed' && (
                <div className="text-xs text-center text-destructive">
                  Failed
                </div>
              )}
            </div>
            <div className="text-center">
              <p className="text-xs font-medium">{item.title}</p>
              {item.status === 'ready' && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-xs p-1 h-auto"
                  onClick={() => onDownloadItem(item.id)}
                >
                  <Download size={12} className="mr-1" />
                  Download
                </Button>
              )}
            </div>
          </div>
        ))}
        
        {/* Add New Take */}
        <div className="flex-shrink-0 space-y-2">
          <div className="filmstrip-item border-dashed border-2 border-muted-foreground/30">
            <div className="text-center">
              <div className="w-6 h-6 mx-auto mb-1 rounded-full border-2 border-dashed border-muted-foreground/50 flex items-center justify-center">
                <span className="text-xs">+</span>
              </div>
              <span className="text-xs">New Take</span>
            </div>
          </div>
          <p className="text-xs text-center text-muted-foreground">Take {items.length + 1}</p>
        </div>
      </div>
    </div>
  );
};