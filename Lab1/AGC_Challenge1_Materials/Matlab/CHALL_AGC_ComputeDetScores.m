function [FD_score, scoresSTR] = CHALL_AGC_ComputeDetScores(...
    DetectionSTR, AGC_Challenge1_STR, show_figures)
%
% Compute face detection score
%
% [FD_score, scoresSTR] = CHALL_AGC_ComputeDetScores(...
%    DetectionSTR, AGC_Challenge1_STR, show_figures)
%
% INPUTS
%   - DetectionSTR: A structure with the results of the automatic detection
%   algorithm, with one element per input image containing field
%   'det_faces'. This field contains as many 4-column rows as faces
%   returned by the detector, each specifying a bounding box coordinates 
%   as [x1,y1,x2,y2], with x1 < x2 and y1 < y2.
%
%   - 9: The ground truth structure (e.g.
%   AGC_Challenge1_TRAINING or AGC_Challenge1_TEST).
%
%   - show_figures: A flag to enable detailed displaying of the results for
%   each input image. If set to zero it just conputes the scores, with no
%   additional displaying.
%
% OUTPUTS
%   - FD_score:     The final detection score obtained by the detector
%   - scoresSTR:    Structure with additional detection information 
% 
% --------------------------------------------------------------------
% AGC Challenge 
% Universitat Pompeu Fabra
%

scoresSTR = struct();
for j = 1 : length( AGC_Challenge1_STR )
    if show_figures
        A = imread( AGC_Challenge1_STR(j).imageName );    
        
        clf;
        imshow( A, 'InitialMagnification', 'fit');
        hold on;
        
        for k1 = 1 : size( AGC_Challenge1_STR(j).faceBox, 1 )
            fb = AGC_Challenge1_STR(j).faceBox( k1, : );
            plot( fb([1 1 3 3 1]), fb([2 4 4 2 2]), 'b',...
                'LineWidth', 2);
        end
        for k2 = 1 : size( DetectionSTR(j).det_faces, 1 )
            fdet = DetectionSTR(j).det_faces( k2, : );
            plot( fdet([1 1 3 3 1]), fdet([2 4 4 2 2]), 'g',...
                'LineWidth', 2);
        end
    end
    
    n_actualFaces = size( AGC_Challenge1_STR(j).faceBox, 1 );
    n_detectedFaces = size( DetectionSTR(j).det_faces, 1 );
    if not( n_actualFaces )
        if n_detectedFaces
            scoresSTR(j).F1 = zeros(1, n_detectedFaces);
        else
            scoresSTR(j).F1 = 1;
        end
    else
        if not( n_detectedFaces )
            scoresSTR(j).F1 = zeros( 1, n_actualFaces );
        else
            % Compute all pair-wise scores
            scoresSTR(j).Fmatrix = zeros( n_actualFaces, n_detectedFaces );
            
            for k1 = 1 : n_actualFaces
                f = AGC_Challenge1_STR(j).faceBox(k1, :);
                
                for k2 = 1 : n_detectedFaces
                    g = DetectionSTR(j).det_faces( k2, : );
                    
                    % Intersection box
                    x1 = max( f(1), g(1) );
                    y1 = max( f(2), g(2) );
                    x2 = min( f(3), g(3) );
                    y2 = min( f(4), g(4) );
                    
                    % Areas
                    int_Area = max(0, (x2-x1)) * max(0, (y2-y1));
                    total_Area = (f(3)-f(1)) * (f(4)-f(2)) + ...
                        (g(3)-g(1)) * (g(4)-g(2)) - int_Area;
                    scoresSTR(j).Fmatrix(k1,k2) = int_Area / total_Area;
                    
                end
            end
            
            % Compute the resulting F1 scores
            scoresSTR(j).F1 = zeros(1, ...
                max( n_detectedFaces, n_actualFaces ));
            
            for k3 = 1 : min( n_actualFaces, n_detectedFaces )
                
                % Get the maximum F-score
                max_F = max( scoresSTR(j).Fmatrix(:) );
                [i1, i2] = find( scoresSTR(j).Fmatrix == max_F );
                
                % The detection with the highest F-score is kept
                scoresSTR(j).F1( i2 ) = max_F;
                
                % The actual and detected faces just used cannot be 
                % selected anymore and are therefore voided
                scoresSTR(j).Fmatrix( i1, : ) = 0;
                scoresSTR(j).Fmatrix( :, i2 ) = 0;
                
            end
        end
    end


    if show_figures
        xlabel( sprintf('%.2f  ', scoresSTR(j).F1),...
            'FontSize', 14);
        getframe;
        pause;
    end
end


all_scores = [scoresSTR(:).F1];
FD_score = mean( all_scores );


