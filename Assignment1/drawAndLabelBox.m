function drawAndLabelBox( x,y,templateIndex , dimensions )

startX = round(x - dimensions(templateIndex).width/2 );
endX = round(x + dimensions(templateIndex).width/2 );

startY = round(y - dimensions(templateIndex).height/2 );
endY = round(y + dimensions(templateIndex).height/2 );

lineX = [ startX, endX , endX , startX , startX ] ;
lineY = [ startY , startY, endY , endY , startY ] ;

line( lineX , lineY, 'Color', [0 0 1], 'LineWidth', 3 ) ;
hold on
rmdr = rem(templateIndex,10)-1 ;
digitToShow = rmdr ;
if( digitToShow == -1 )
    digitToShow = 9 ;
end

text( startX , startY , num2str(digitToShow) , 'Color', 'yellow') ;
