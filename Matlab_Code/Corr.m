clear;
clc;

InitTimerVal = tic;
for height = 5:10:65
    for width = 5:10:65
        height
        width
        tic
        data = zeros(width,height);
        for j = 1:1:height
            for i = 1:1:width
                data(j,i) = 1;
            end
        end

        padded = zeros(width*2-1, height*2-1);
        for k = 1:1:height
            for l = 1:1:width
                padded(k,l) = data(k,l);
            end
        end

        Y = fft2(padded);

        Z = abs(Y.*conj(Y));

        out = ifft2(Z);

        TL = out(width+1:width*2-1,height+1:height*2-1);
        TR = out(width+1:width*2-1,1:height);
        BR = out(1:width,1:height);
        BL = out(1:width,height+1:height*2-1);

        final = [TL,TR;BL,BR];
        
        timerVal = toc;
        disp(timerVal*1000)

        %R = xcorr2(data,data);

        %compare = abs(final-R)
    end
end
