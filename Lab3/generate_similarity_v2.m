function simScores = generate_similarity_v2(img_path, file_name, simScores)
% GENERATE_SIMILARITY
% simScores = generate_similarity(img_path, file_name )
%
% or, if the annotation was interrupted at half-way (to be able to
% continue):
%
% simScores = generate_similarity(img_path, file_name, simScores )
%
%   inputs: - img_path: location of the images
%           - file_name: file to save your scores (use .mat extension)
%           - simScores; if you need to stop your annotation before you
%           finish, you can continue from the point you left by re-entering
%           the output of the funcation as the last parameter.
%
%   outputs: - simScores: the stored similarity values that you entered
%
%Scores from 0 to 9.



images = dir([img_path, '/*.JPG']);

if(exist('simScores', 'var'))
    similarityM = simScores.similarityM;
    consistencyM = simScores.consistencyM;
else
    similarityM = Inf(length(images), length(images));
    consistencyM = Inf(length(images), length(images));
    simScores.similarityM = similarityM;
    simScores.consistencyM = consistencyM;
end

strmessage_1 = '[0,1] - Not similar at all';
strmessage_2 = '[2,3] - Not very similar';
strmessage_3 = '[4,5] - +/- or do not know';
strmessage_4 = '[6,7] - Quite similar';
strmessage_5 = '[8,9] - The same';
prompt = sprintf('%s\n%s\n%s\n%s\n%s',...
    strmessage_1, strmessage_2, strmessage_3, strmessage_4, strmessage_5);

InputFig = figure;
UppDiagonal = triu(ones(length(images), length(images)),1);
while (sum(isinf(similarityM(UppDiagonal == 1))) > 0)
    [X,Y] = meshgrid(1:length(images),1:length(images));
    r_idx_x = randperm( length( X(:)) );
    X = reshape( X( r_idx_x ), [length( images ), length( images )]);
    r_idx_y = randperm( length( Y(:)) );
    Y = reshape( Y( r_idx_y ), [length( images ), length( images )]);
    %         X = Shuffle(X, 2);
    %         Y = Shuffle(Y, 1);

    for i = 1:length(images)
        for j = 1:length(images)
            if(X(i,j) == Y(i,j))
                similarityM(X(i,j),Y(i,j)) = 9;
                continue
            elseif (isinf(similarityM(X(i,j),Y(i,j))))
                counter = sum(UppDiagonal(:)) - sum(isinf(similarityM(UppDiagonal == 1)));
                t = get(0,'MonitorPosition');
                im1F = imread([img_path, '/', images(X(i,j)).name])*1.2;
                clf
                set(gcf,'color','w')
                set(InputFig, 'Position', t(1,:) + [0 22 0 0])                
                im2F = imread([img_path, '/', images(Y(i,j)).name])*1.2;
                subplot(1,2,1)
                imshow(im1F, [0 255])
                subplot(1,2,2)
                imshow(im2F, [0 255])
                
                ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0  1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
                text(0.2, 0.95,['\fontsize{30}How similar are these emotions? ', num2str(counter)]);
                %title(['\fontsize{30}How similar are these emotions? ', )

                
                boxtitle = 'Value: ';
                answer = inputdlg(prompt, boxtitle);
                if(isempty(answer))
                    close all
                    return
                end
                integerValue = str2num(answer{1});
                while (isempty(integerValue) || integerValue > 9) % Check for a valid integer.
                    answer = inputdlg(prompt, boxtitle);
                    if(isempty(answer))
                        close all
                        return
                    end
                    integerValue = str2num(answer{1});
                end
                similarityM(X(i,j),Y(i,j)) = integerValue;
                similarityM(Y(i,j),X(i,j)) = similarityM(X(i,j),Y(i,j));
                simScores.similarityM = similarityM;
                save(file_name, 'simScores')
            end
        end
    end
end
[X,Y] = meshgrid(1:length(images),1:length(images));
r_idx_x = randperm( length( X(:)) );
X = reshape( X( r_idx_x ), [length( images ), length( images )]);
r_idx_y = randperm( length( Y(:)) );
Y = reshape( Y( r_idx_y ), [length( images ), length( images )]);
%         X = Shuffle(X, 2);
%         Y = Shuffle(Y, 1);
if(~exist('counter', 'var'))
    counter = (length(images)^2 - length(images))/2;
end
consistencyMcounter = 0;
while(consistencyMcounter ~= ((length(images)^2 - length(images))/2 + length(images)))
    i = floor(1 + (length(images)-1).*rand(1, 1));
    j = floor(1 + (length(images)-1).*rand(1, 1));
    
    if(X(i,j) == Y(i,j))
        continue
    elseif (isinf(consistencyM(X(i,j),Y(i,j))))
        consistencyMcounter = counter + sum(UppDiagonal(:)) - sum(isinf(consistencyM(UppDiagonal == 1)));
        t = get(0,'MonitorPosition');
        im1F = imread([img_path, '/', images(X(i,j)).name])*1.2;
        clf
        set(gcf,'color','w')
        set(InputFig, 'Position', t(1,:) + [0 22 0 0])
        im2F = imread([img_path, '/', images(Y(i,j)).name])*1.2;
        subplot(1,2,1)
        imshow(im1F, [0 255])
        subplot(1,2,2)
        imshow(im2F, [0 255])
        ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0  1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
        text(0.2, 0.95,['\fontsize{30}How similar are these emotions? ', num2str(consistencyMcounter)]);
        boxtitle = 'Value: ';
        answer = inputdlg(prompt, boxtitle);
        if(isempty(answer))
            close all
            return
        end
        integerValue = str2num(answer{1});
        while (isempty(integerValue) || integerValue > 9) % Check for a valid integer.
            answer = inputdlg(prompt, boxtitle);
            if(isempty(answer))
                close all
                return
            end
            integerValue = str2num(answer{1});
        end
        consistencyM(X(i,j),Y(i,j)) = integerValue;
        consistencyM(Y(i,j),X(i,j)) = consistencyM(X(i,j),Y(i,j));
        simScores.consistencyM = consistencyM;
        save(file_name, 'simScores')
    end
end
close all
end