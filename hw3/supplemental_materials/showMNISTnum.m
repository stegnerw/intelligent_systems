%Reads the text version of the 5000-number MNIST data file and displays the
%first 100 in a 10X10 array.
%
%Written by: Ali Minai(3/14/16)

U = load('MNISTnumImages5000_balanced.txt');

for i=1:10
    for j = 1001:1010
        v = reshape(U((i-1)*10+j,:),28,28);
        subplot(10,10,(i-1)*10+j-1000)
        image(64*v)
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end
