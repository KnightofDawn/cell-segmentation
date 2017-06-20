#!/usr/bin/python
import numpy as np
import cv2
import argparse as ap
import time
from matplotlib import pyplot as plt

# Muestra una imagen en una ventana
def display(tag, im, scale=0.35):
	if args.interactive:
		res = cv2.resize(im, None, fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
		cv2.imshow(tag, res)

# Produce el histograma de una imagen
def hist_lines(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print("hist_lines applicable only for grayscale images")
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y

# Permite visualizar una imagen binaria sobre otra en color como una máscara con un tono
def overlayHue(im,mask,hue):
    imask = cv2.bitwise_not(mask) # máscara invertida
    overlay = np.ones(mask.shape, dtype=np.uint8) * int(hue) # array inicializado al tono deseado
    overlay = cv2.bitwise_and(overlay,overlay,mask=mask) # aplicamos máscara
    imHSV = cv2.cvtColor(im,cv2.COLOR_BGR2HSV) # convertimos a HSV la imagen
    bgHue = cv2.bitwise_and(imHSV[:,:,0],imHSV[:,:,0],mask=imask) # máscara inversa al canal tono
    imHSV[:,:,0] = cv2.add(bgHue,overlay) # sumamos
    return cv2.cvtColor(imHSV,cv2.COLOR_HSV2BGR) # y tenemos un overlay del tono deseado

# Segmenta las células en la imagen
def segment_cells(im):
	print('Iniciando segmentación celular...')
	
	# Convertir a HSV
	hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	eqs = cv2.equalizeHist(s)
	display('HSV', hsv, 0.15)
	display('Canal S ecualizado', eqs, 0.35)
	display('Histograma', hist_lines(eqs), 1)

	# Gaussiana para reducir ruido
	gauss = cv2.GaussianBlur(eqs,(51,51),9)
	display('Gauss', gauss, 0.35)

	# Umbralizar canal S
	thresh = cv2.adaptiveThreshold(gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 801, -10)
	display('Umbralización S', thresh, 0.2)

	# Apertura para eliminar ruido
	opkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(75,75))
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, opkernel)
	display('Apertura', opening, 0.2)
	display('Apertura overlay', overlayHue(im, opening, 85), 0.35)

	# Determinación de areas de certeza
	erkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
	sure_bg = cv2.dilate(opening, erkernel, iterations=2)
	display('Fondo seguro', overlayHue(im, sure_bg, 0), 0.35)

	amount, labels = cv2.connectedComponents(opening)
	display('Connected components', cv2.applyColorMap(np.uint8(255*(labels / amount)), cv2.COLORMAP_JET), 0.35) # presentación de componentes conexas con colormap

	# Separamos cada componente conectada a un array para aplicar recorte de transformada de distancia por separado (en función del tamaño del clúster)
	split = [ np.uint8(255*(labels==i+1)) for i in range(amount-1) ]

	merge = np.zeros_like(opening)
	for comp in split:
		dist_transform = cv2.distanceTransform(comp, cv2.DIST_L2, 5) # transformada de distancia de cada componente
		ret, thresh = cv2.threshold(dist_transform, 0.45*dist_transform.max(), 255, 0) # umbralización al 35% de la distancia máxima de ese componente
		merge += np.uint8(thresh) # lo juntamos en el original

	# Apertura para eliminar detecciones espurias
	opkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
	opening = cv2.morphologyEx(merge, cv2.MORPH_OPEN, opkernel)
	display('Apertura del merge', overlayHue(im, opening, 25), 0.35)

	# Determinación area incertidumbre
	sure_fg = np.uint8(opening) 
	unknown = cv2.subtract(sure_bg,sure_fg)
	display('Area incertidumbre', unknown, 0.15)

	# Filtrado de Gauss de la original para aplicar watershed
	color_gauss = cv2.GaussianBlur(im,(51,51),9)

	# Llenado de cuencas
	markers = cv2.connectedComponents(sure_fg)[1]
	markers = markers+1
	markers[unknown==255] = 0
	markers = cv2.watershed(color_gauss, markers) # aplicamos watershed sobre la versión con filtrado gaussiano de la original en color

	# Resultados
	result = np.array(im, copy=True)
	result[markers==-1] = [0,255,255]
	display('Watershed', result, 0.35)
	
	return markers

# Determina si un marcador es rojo o azul en la imagen dada, devuelve verdadero si es rojo
# im - imagen original
# mark - máscara con el marcador segmentado
def classify(im, mark):
	BLUE = 125
	RED = 150

	hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	
	medianHue = np.average(h[mark==255])
	##print('Mediana tono: ' + str(medianHue))
	blueDist = abs(medianHue - BLUE)
	redDist = abs(medianHue - RED)
	
	return blueDist > redDist

# Segmenta los marcadores celulares en la imagen y los clasifica en rojos o azules
def find_markers(im):
	print('Iniciando segmentación de marcadores...')
	# Convertir a escala de grises
	grayscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	display('Escala de grises', grayscale)
	display('Original histogram', hist_lines(grayscale), 1)

	# Filtro top-hat negro en escala de grises
	bhkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
	blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, bhkernel)
	display('Blackhat', blackhat)
		
	# Umbralización
	ret, thresh = cv2.threshold(blackhat,12,255,cv2.THRESH_BINARY)
	display('Umbralización', thresh)
	display('Umbralización sobre original', overlayHue(im, thresh, 25))

	# Detección de componentes conexas
	amount, labels = cv2.connectedComponents(thresh)
	display('Componentes conexas (' + str(amount) + ')', cv2.applyColorMap(np.uint8(255*(labels / amount)), cv2.COLORMAP_JET)) # presentación de componentes conexas con colormap

	print('Iniciando clasificación de marcadores...')
	split = [ np.uint8(255*(labels==i+1)) for i in range(amount-1) ]

	red, blue = np.zeros_like(thresh), np.zeros_like(thresh)
	
	for mark in split:
		if classify(im, mark):
			red += mark
		else:
			blue += mark

	amountr, ccred = cv2.connectedComponents(red)
	display('Componentes conexas rojos (' + str(amountr) + ')', cv2.applyColorMap(np.uint8(255*(ccred / amountr)), cv2.COLORMAP_JET))
	amountb, ccblue = cv2.connectedComponents(blue)
	display('Componentes conexas azules (' + str(amountb) + ')', cv2.applyColorMap(np.uint8(255*(ccblue / amountb)), cv2.COLORMAP_JET))
	return ccred, ccblue

def count(cells, markers):
	return [ np.count_nonzero(count) for count in [np.unique(markers[cell]) for cell in [ cells==i+1 for i in range(np.max(cells)) ]]]

def report(path, time, cellnum, countRed, countBlue):
	print('El procesamiento de ' + path + ' ha concluído en {:.1f} segundos.'.format(time))
	print('Informe detallado, células numeradas en sentido de lectura en la imagen (de izquierda a derecha y de arriba a abajo):')
	print('# célula'.ljust(8), 'rojos'.ljust(8), 'azules'.ljust(8))
	for i in range(cellnum):
		print(str(i+1).ljust(8), str(countRed[i+1]).ljust(8), str(countBlue[i+1]).ljust(8))
	
	reportText = ('Se encontraron {} células.\n'
		'Media y desviación típica de marcadores rojos por célula: {:.3f}, {:.3f}\n'
		'Media y desviación típica de marcadores azules por célula: {:.3f}, {:.3f}')
	
	redAvg = np.average(countRed)
	redStd = np.std(countRed)
	blueAvg = np.average(countBlue)
	blueStd = np.std(countBlue)

	print(reportText.format(cellnum, redAvg, redStd, blueAvg, blueStd))
	print('Marcadores encontrados fuera de las células: {} rojos, {} azules'.format(countRed[0], countBlue[0]))
	print('')

def process(path):
	print('Procesando ' + path)
	start = time.time()
	# Cargar imagen
	original = cv2.resize(cv2.imread(path,1), (0,0), fx=0.5, fy=0.5)
	display('Original', original)
	
	cells = segment_cells(original)
	cellnum = np.max(cells)-1
	red, blue = find_markers(original)
	countRed, countBlue = count(cells, red), count(cells, blue)
	end = time.time()
	report(path, end-start, cellnum, countRed, countBlue)

parser = ap.ArgumentParser(description='Realiza recuentos de marcadores en imágenes de células mediante técnicas de Visión Artificial')
parser.add_argument('imagenes', help='la(s) imagen(es) que se va a procesar separadas por espacios', type=str, nargs='+')
parser.add_argument('-i', '--interactive', dest='interactive', action='store_const', const=True, default=False, help='si está presente, se muestra cada una de las etapas de procesamiento en ventanas',)

args = parser.parse_args()

for path in args.imagenes:
	process(path)

if args.interactive:
	while True:
		k = cv2.waitKey(0)
		if k == 27:
		    cv2.destroyAllWindows()
		    break

